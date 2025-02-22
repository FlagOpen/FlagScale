#!/bin/bash

echo "The current directory is: $(pwd)"

# Function to execute a command and handle failures
run_command() {
  eval $1
  if [ $? -ne 0 ]; then
    echo "Command failed: $1"
    return 1
  fi
  return 0
}

source tests/scripts/_gpu_check.sh

# Path to the YAML configuration file
CONFIG_FILE="tests/scripts/functional_tests/config.yml"

# Function to parse the configuration file and run tests
test_model() {
  local _type=$1
  local _model=$2
  # Use parse_config.py to parse the YAML file with test type and test model
  local _cases=$(python tests/scripts/functional_tests/parse_config.py --config $CONFIG_FILE --type $_type --model $_model)

  # Convert the parsed test cases to an array
  IFS=' ' read -r -a _cases <<< "$_cases"

  # Check if _cases is not an empty list
  if [ ${#_cases[@]} -eq 0 ]; then
    echo "No test cases found for model '$_model' with test type '$_type'. Exiting."
    exit 0
  fi

  # Loop through each test case, remove leading '-', and run the test
  for _case in "${_cases[@]}"; do
    # Remove leading '-'
    _case=${_case#-}

    test_times=1
    # Attempt to run the test 5 times
    for attempt_i in {$test_times}; do
      wait_for_gpu

      echo "---------"
      echo "Attempt $attempt_i for model ${_model} with type ${_type} and case: ${_case}"
      echo "---------"

      # Remove previous results if exist
      result_path="tests/functional_tests/test_cases/${_type}/${_model}/results_test/${_case}"
      if [ -d $result_path ]; then
        rm -r $result_path
      fi

      if [ ${_type} = "serve" ]; then
        if [ ${attempt_i} -gt 1 ]; then
          break
        fi

        # Function to execute a command and clear serve process
        clear_serve() {
          ray stop
          pkill -f "python"
          return 0
        }

        # Function to print serve logs
        print_log() {
          local log_file=$1
          echo "------------------ serve log begin -----------------------"
          if [[ -n "$log_file" && -f "$log_file" ]]; then
            echo "Log file found at $log_file. Printing log content:"
            cat "$log_file"
          else
            echo "No log file found at $log_file or path is empty."
          fi
          echo "------------------ env ----------------------"
          env
          pip list
          echo "------------------ serve log end   -----------------------"
        }

        log_path="./outputs/${_case}/serve_logs/host_0_localhost.output"

        export no_proxy="127.0.0.1,localhost"
        # serve
        echo "python run.py --config-path tests/functional_tests/test_cases/${_type}/${_model}/conf --config-name ${_case} action=run"
        run_command "python run.py --config-path tests/functional_tests/test_cases/${_type}/${_model}/conf --config-name ${_case} action=run"
        if [ $? -ne 0 ]; then
          echo "Test failed on attempt $attempt_i for serve case $_case."
          print_log "$log_path"
          clear_serve
          exit 1
        fi
        sleep 2m
        # call
        echo "python tests/functional_tests/test_cases/${_type}/${_model}/test_call.py"
        run_command "python tests/functional_tests/test_cases/${_type}/${_model}/test_call.py"
        if [ $? -ne 0 ]; then
          echo "Test failed on attempt $attempt_i for call serve case $_case."
          print_log "$log_path"
          clear_serve
          exit 1
        fi
        #clear
        clear_serve
      else

        run_command "python run.py --config-path tests/functional_tests/test_cases/${_type}/${_model}/conf --config-name ${_case} action=test"
        if [ $? -ne 0 ]; then
          echo "Test failed on attempt $attempt_i for case $_case."
          exit 1
        fi

        run_command "pytest -p no:warnings -s tests/functional_tests/test_utils/test_equal.py --test_path=tests/functional_tests/test_cases --test_type=${_type} --test_model=${_model} --test_case=${_case}"
        if [ $? -ne 0 ]; then
          echo "Pytest failed on attempt $attempt_i for case $_case."
          exit 1
        fi
      fi

      # Ensure that pytest check is completed before deleting the folder
      sleep 10s
    done
    echo "All $test_times attempts successful for case $_case for model ${_model}."
  done
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --type) type="$2"; shift ;;
        --model) model="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate that the required parameters --type and --model are provided
if [ -z "$type" ]; then
  echo "Error: --type is required"
  exit 1
fi

if [ -z "$model" ]; then
  echo "Error: --model is required"
  exit 1
fi

# Run the tests based on the provided test type and test model
test_model "$type" "$model"
