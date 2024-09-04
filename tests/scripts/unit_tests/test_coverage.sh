#!/bin/bash

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --backend) backend="$2"; shift ;;
        --id) id="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure necessary arguments are provided
if [ -z "$backend" ]; then
    echo "Usage: $0 --backend <backend> [--id <id>]"
    exit 1
fi

# Set the default value of id to 0
if [ -z "$id" ]; then
    id=0
fi

# Define the coverage report directory and the coverage XML report file
report_dir="/workspace/report/$id/cov-report-${backend}"
coverage_file="coverage.xml"

# Check if the coverage XML file exists
if [ ! -f "$report_dir/$coverage_file" ]; then
    echo "Error: Coverage file $report_dir/$coverage_file not found!"
    exit 1
fi

# Add the current working directory to the list of safe directories in Git
git config --global --add safe.directory /__w/FlagScale/FlagScale

# Ensure the upstream main branch is up-to-date
git remote add upstream https://github.com/FlagOpen/FlagScale.git
git fetch upstream main:main

# Get the latest common ancestor between the current branch and upstream main
common_ancestor=$(git merge-base HEAD upstream/main)

# Check the coverage for the new code changes against the common ancestor
echo "Checking coverage for the new code changes..."
diff-cover "$report_dir/$coverage_file" --compare-branch=$common_ancestor --fail-under=70

# If diff-cover exits with a non-zero status, it means the coverage is below 70%
if [ $? -ne 0 ]; then
    echo "Test coverage for new code is below 70% in the $backend. Please add more tests."
    exit 1
else
    echo "Test coverage for new code meets the 70% requirement in the $backend."
fi
