#!/bin/bash

# Run each command and capture its return value
commands=(
    # For train
    "tests/scripts/functional_tests/test_task.sh --type train --task aquila"
    # TODO: need fix
    # "tests/scripts/functional_tests/test_task.sh --type train --task deepseek"
    # "tests/scripts/functional_tests/test_task.sh --type train --task mixtral"
    # "tests/scripts/functional_tests/test_task.sh --type train --task llava_onevision"
    # For hetero-train
    # "tests/scripts/functional_tests/test_task.sh --type hetero_train --task aquila"
    # For inference
    "tests/scripts/functional_tests/test_task.sh --type inference --task deepseek"
    "tests/scripts/functional_tests/test_task.sh --type inference --task qwen3"
    "tests/scripts/functional_tests/test_task.sh --type inference --task deepseek_flaggems"
    "tests/scripts/functional_tests/test_task.sh --type inference --task qwen3_flaggems"
    # For inference-pipeline
    "tests/scripts/functional_tests/test_task.sh --type inference-pipeline --task Qwen3-4B"
    "tests/scripts/functional_tests/test_task.sh --type inference-pipeline --task Qwen3-4B --flaggems enable"
    # For inference-pipeline: other hardware
    # "tests/scripts/functional_tests/test_task.sh --type inference-pipeline --task Qwen3-4B --hardware bi_v150"
    # "tests/scripts/functional_tests/test_task.sh --type inference-pipeline --task Qwen3-4B --hardware bi_v150 --flaggems enable"
    # "tests/scripts/functional_tests/test_task.sh --type inference-pipeline --task Qwen3-4B --hardware cambricon_mlu"
    # "tests/scripts/functional_tests/test_task.sh --type inference-pipeline --task Qwen3-4B --hardware cambricon_mlu --flaggems enable"
    # For serve
    # "tests/scripts/functional_tests/test_task.sh --type serve --task base"
)

if [ ! -f tests/functional_runtime.txt ];then
    touch tests/functional_runtime.txt
else
    echo "" > tests/functional_runtime.txt
fi

echo "start time: $(date +"%Y-%m-%d %H:%M:%S")" >> tests/functional_runtime.txt
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
echo "end time: $(date +"%Y-%m-%d %H:%M:%S")" >> tests/functional_runtime.txt

python tests/scripts/test.time_statistics.py functional