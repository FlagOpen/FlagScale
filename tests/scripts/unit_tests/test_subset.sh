#!/bin/bash

echo "The current directory is: $(pwd)"

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
echo "config:" $config

# Split the Python output and set default values
set_environment=$(echo $config | cut -d '|' -f 1)
root=$(echo $config | cut -d '|' -f 2);root=${root:-"tests/unit_tests"}
coverage=$(echo $config | cut -d '|' -f 3)
type=$(echo $config | cut -d '|' -f 4);type=${type:-"batch"}
depth=$(echo $config | cut -d '|' -f 5);depth=${depth:-"all"}
ignore=$(echo $config | cut -d '|' -f 6)
deselect=$(echo $config | cut -d '|' -f 7)
id=${id:-0}

# Set the test path based on root and subset
if [[ "$subset" =~ [./]$ ]]; then
    subset="${subset##*[./]}"
fi
path="${root}/${subset}"
path="${path%/}"

# Output the test name
echo "Running test: $backend -> ${subset:-./}"

# Execute the set_environment commands
eval "$set_environment"

# Function to check if both reports are complete
check_reports_complete() {
    local xml_report="$1"
    local html_report="$2"

    local xml_previous_size=0
    local html_previous_size=0
    local xml_current_size
    local html_current_size

    # Continuously check size stability for both reports
    while true; do
        xml_current_size=$(stat --format=%s "$xml_report" 2>/dev/null || echo 0)
        html_current_size=$(stat --format=%s "$html_report" 2>/dev/null || echo 0)

        if [ "$xml_previous_size" -eq "$xml_current_size" ] && [ "$html_previous_size" -eq "$html_current_size" ]; then
            break
        fi

        xml_previous_size=$xml_current_size
        html_previous_size=$html_current_size
        sleep 5s
    done
}

# Function to run tests (either single or batch)
run_tests_case() {
    local _test_files="$1"
    local _backend="$2"
    local _coverage="$3"
    local _xml_report="$4"
    local _html_report="$5"
    local _ignore_cmd="$6"
    local _deselect_cmd="$7"

    wait_for_gpu

    if [[ -n "$_test_files" ]]; then
        for _test_file in $_test_files; do
            echo "Running test: ${_test_file}"
            tests_case = "torchrun --nproc_per_node=8 -m pytest --cov=${_backend}/${_coverage} --cov-append --cov-report=xml:${_xml_report} --cov-report=html:${_html_report} -q -x -p no:warnings ${_ignore_cmd} ${_deselect_cmd} ${_test_file}"
            echo "$tests_case"
            eval "$tests_case"

            # Check the exit status of pytest
            if [ $? -ne 0 ]; then
                echo "Test failed: ${_test_file}"
                exit 1
            fi

            # Check if both report files are complete
            check_reports_complete "${_xml_report}" "${_html_report}"

            if [ $? -ne 0 ]; then
                echo "Check reports failed: ${_xml_report} ${_html_report}"
                exit 1
            fi
        done
    fi
}

# Function to run tests at a specific depth
run_tests() {
    local _type="$1"
    local _path="$2"
    local _depth="$3"
    local _ignore="$4"
    local _deselect="$5"

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
        for item in $_ignore; do
            _clean_item=${item#-}
            _clean_item=$(echo "$_clean_item" | tr -d "',[]")
            ignore_cmd+="--ignore=${path}/${_clean_item} "
            _test_files=$(echo "$_test_files" | grep -v "${_path}/${_clean_item}")
        done
    fi

    _test_files=$(echo "$_test_files" | tr '\n' ' ')

    # Process the raw deselect into bash-friendly --deselect parameters
    deselect_cmd=""
    if [ -n "$_deselect" ]; then
        for item in $_deselect; do
            _clean_item=${item#-}
            _clean_item=$(echo "$_clean_item" | tr -d "',[]")
            deselect_cmd+="--deselect=${path}/${_clean_item} "
        done
    fi

    local xml_report="/workspace/report/$id/cov-report-${backend}/coverage.xml"
    local html_report="/workspace/report/$id/cov-report-${backend}"
    export COMMIT_ID=$id

    echo "************************"
    echo $(which python)
    echo $(which conda)
    echo "************************"

    run_tests_case "$_test_files" "$backend" "$coverage" "$xml_report" "$html_report" "$ignore_cmd" "$deselect_cmd"
}

# Run tests based on type, path, and depth
run_tests "$type" "$path" "$depth" "$ignore" "$deselect"
