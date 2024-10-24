#!/bin/bash

# Run each command and capture its return value
commands=(
    "tests/scripts/functional_tests/test_model.sh --type train --model aquila"
    "tests/scripts/functional_tests/test_model.sh --type train --model mixtral"
    # for hetero-train
    "tests/scripts/functional_tests/test_model.sh --type hetero_train --model aquila"
    # Add in the feature
    # "tests/scripts/functional_tests/test_model.sh --type inference --model vllm"
)


for cmd in "${commands[@]}"; do
    attempts=0
    max_attempts=10
    success=0

    while [ $attempts -lt $max_attempts ]; do
        echo "Attempt $((attempts + 1)) of $max_attempts"
        
        # Execute the command
        $cmd
        # Capture the return value
        return_value=$?

        # Check if the return value is zero (success)
        if [ $return_value -eq 0 ]; then
            echo "Command '$cmd' succeeded on attempt $((attempts + 1))"
            success=1
            break
        else
            echo "Warning: Command '$cmd' failed on attempt $((attempts + 1))"
            attempts=$((attempts + 1))
        fi
    done

    # If the command failed after the maximum attempts, exit with an error
    if [ $success -eq 0 ]; then
        echo "Error: Command '$cmd' failed after $max_attempts attempts"
        exit 1
    fi
done