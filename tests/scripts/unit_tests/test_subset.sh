#!/bin/bash

source tests/scripts/_gpu_check.sh

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --backend) backend="$2"; shift ;;
        --subset) subset="$2"; shift ;;
        --id) id="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure necessary arguments are provided
if [ -z "$backend" ] || [ -z "$subset" ]; then
    echo "Usage: $0 --backend <backend> --subset <subset> [--id <id>]"
    exit 1
fi

# Configuration file path
config_file="tests/scripts/unit_tests/config.yml"
if [ ! -f "$config_file" ]; then
    echo "Configuration file $config_file not found!"
    exit 1
fi

# Call the Python script to extract the relevant configuration
config=$(python3 tests/scripts/unit_tests/parse_config.py "$config_file" "$backend" "$subset")

# Split the Python output
set_environment=$(echo $config | cut -d '|' -f 1)
root=$(echo $config | cut -d '|' -f 2)
coverage=$(echo $config | cut -d '|' -f 3)
type=$(echo $config | cut -d '|' -f 4)
depth=$(echo $config | cut -d '|' -f 5)
ignore=$(echo $config | cut -d '|' -f 6)

# Set default value
if [ -z "$root" ]; then
    root="tests/unit_tests"
fi

if [ -z "$type" ]; then
    type="batch"
fi

if [ -z "$depth" ]; then
    depth="all"
fi

if [ -z "$id" ]; then
    id=0
fi

# Set the test path based on root and subset
if [[ "$subset" =~ [./]$ ]]; then
    subset="${subset##*[./]}"
fi
path="${root}"/"${subset}"
path="${path%/}"

# Output the test name
echo "Running test: $backend -> ${subset:-./}"

# Execute the set_environment commands
eval "$set_environment"

# Function to run tests at a specific depth
run_tests() {
    local _type="$1"
    local _path="$2"
    local _depth="$3"
    local _ignore="$4"

    local _test_files="$_path"
    if ! ([ "$_depth" = "all" ] && [ "$_type" = "batch" ]); then
        if [ "$_depth" = "all" ]; then
            _depth=$(find "$_path" -type d | awk -F/ '{print NF-1}' | sort -nr | head -n 1)
        fi
        _test_files=$(find "$_path" -mindepth 1 -maxdepth $_depth -type f -name "test_*.py" | sort)
    fi

    # Process the raw ignore into bash-friendly --ignore parameters
    ignore_cmd=""
    if [ -n "$_ignore" ]; then
        # Handle the entire string as a list of files
        for item in $_ignore; do
            # Remove the leading '-' from each item if it exists
            _clean_item=${item#-}
            ignore_cmd+="--ignore=${path}/${_clean_item} "
            _test_files=$(echo "$_test_files" | grep -v "${_path}/${_clean_item}")
        done
    fi

    _test_files=$(echo "$_test_files" | tr '\n' ' ')

    if [ "$_type" == "batch" ]; then
        wait_for_gpu
        echo "Running batch test: $_test_files"
        torchrun --nproc_per_node=8 -m pytest --import-mode=importlib --cov=${backend}/${coverage} --cov-append --cov-report=xml:/workspace/report/$id/cov-report-${backend}/coverage.xml --cov-report=html:/workspace/report/$id/cov-report-${backend} -q -x -p no:warnings $ignore_cmd $_test_files
        if [ $? -ne 0 ]; then
            echo "Test failed: $_test_files"
            exit 1
        fi
    elif [ "$_type" == "single" ]; then
        for _test_file in $_test_files; do
            wait_for_gpu
            echo "Running single test: $_test_file"
            torchrun --nproc_per_node=8 -m pytest --import-mode=importlib --cov=${backend}/${coverage} --cov-append --cov-report=xml:/workspace/report/$id/cov-report-${backend}/coverage.xml --cov-report=html:/workspace/report/$id/cov-report-${backend} -q -x -p no:warnings $ignore_cmd $_test_file
            # Check the exit status of pytest
            if [ $? -ne 0 ]; then
                echo "Test failed: $_test_file"
                exit 1
            fi
        done
    fi

    # Ensure the test report is generated
    report_directory="/workspace/report/$id/cov-report-${backend}"
    xml_report="$report_directory/coverage.xml"

    # Wait for report generation
    while [ ! -f "$xml_report" ]; do
        echo "Waiting for the test reports to be generated..."
        sleep 5
    done

    echo "Test reports generated successfully."
}

# Run tests based on type, path, and depth
run_tests "$type" "$path" "$depth" "$ignore"