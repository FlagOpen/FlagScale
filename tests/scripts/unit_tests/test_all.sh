#!/bin/bash

# Run each command and capture its return value
commands=(
    # # unit tests -> megatron
    # "rm -rf /workspace/report/0/cov-report-megatron"
    # "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset data"
    # "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset dist_checkpointing"
    # "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset distributed"
    # "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset fusions"
    # "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset inference"
    # "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset models"
    # "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset pipeline_parallel"
    # "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset tensor_parallel"
    # "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset transformer/moe"
    # "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset transformer"
    # "tests/scripts/unit_tests/test_subset.sh --backend megatron --subset ./"
    # # coverage test -> megatron
    # "./tests/scripts/unit_tests/test_coverage.sh --backend megatron"

    # unit tests -> flagscale
    "rm -rf /workspace/report/0/cov-report-flagscale"
    "tests/scripts/unit_tests/test_subset.sh --backend flagscale --subset ./"
    "tests/scripts/unit_tests/test_subset.sh --backend flagscale --subset launcher"
    # coverage test -> flagscale
    "./tests/scripts/unit_tests/test_coverage.sh --backend flagscale"

    # You can add your own test subset here
    # "tests/scripts/unit_tests/test_subset.sh --backend flagscale --subset your_subset"
)

for cmd in "${commands[@]}"; do
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
done
