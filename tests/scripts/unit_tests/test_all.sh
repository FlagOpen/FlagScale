#!/bin/bash
echo "Set fd limit to 65535"
ulimit -n 65535

if [ -d "/workspace/report/0/cov-report-megatron" ]; then rm -r /workspace/report/0/cov-report-megatron; fi
if [ -d "/workspace/report/0/cov-temp-megatron" ]; then rm -r /workspace/report/0/cov-temp-megatron; fi
if [ -d "/workspace/report/0/cov-report-flagscale" ]; then rm -r /workspace/report/0/cov-report-flagscale; fi
if [ -d "/workspace/report/0/cov-temp-flagscale" ]; then rm -r /workspace/report/0/cov-temp-flagscale; fi

# Run each command and capture its return value
commands=(
    # unit tests -> megatron
    "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset data"
    "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset dist_checkpointing"
    "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset distributed"
    "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset export"
    "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset fusions"
    "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset inference"
    "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset models"
    "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset pipeline_parallel"
    "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset post_training"
    "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset ssm"
    "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset tensor_parallel"
    "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset transformer/moe"
    "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset transformer"
    "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset ./"
    # coverage test -> megatron
    "tests/scripts/unit_tests/test_coverage.sh --backend megatron --status offline"

    # unit tests -> flagscale
    "tests/scripts/unit_tests/test_subset.sh --backend flagscale --subset runner"
    "tests/scripts/unit_tests/test_subset.sh --backend flagscale --subset ./"
    # coverage test -> flagscale
    "tests/scripts/unit_tests/test_coverage.sh --backend flagscale --status offline"

    # You can add your own test subset here
    # "tests/scripts/unit_tests/test_subset.sh --backend flagscale --subset your_subset"
)

if [ ! -f tests/unit_runtime.txt ];then
    touch tests/unit_runtime.txt
else
    echo "" > tests/unit_runtime.txt
fi
echo "start time: $(date +"%Y-%m-%d %H:%M:%S")" >> tests/unit_runtime.txt

for cmd in "${commands[@]}"; do

    running_start_time=`date +%s`

    # Execute the command
    $cmd
    # Capture the return value
    return_value=$?
    # Check if the return value is non-zero (an error occurred)
    if [ $return_value -ne 0 ]; then
        # Output the error message and the failing command
        echo "Error: Command '$cmd' failed"
        # Throw an exception by exiting the script with a non-zero status
        exit 1
    fi

    running_end_time=`date +%s`
    echo ">>>: ${cmd} runtime: $((running_end_time-running_start_time))" >> tests/unit_runtime.txt

    echo "Success: Command '$cmd' successed"
done

echo "end time: $(date +"%Y-%m-%d %H:%M:%S")" >> tests/unit_runtime.txt

python tests/scripts/test.time_statistics.py unit