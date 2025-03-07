#!/bin/bash

echo "The current directory is: $(pwd)"

# Function to execute a command and handle failures
run_command() {
  echo "$1"
  eval "$1"
  local command_status=$?  # Capture the command's exit status
  local attempt_i=$2       # Capture the attempt number from the second argument
  local type_name=$3       # Capture the test type name from the third argument
  local mission_name=$4    # Capture the test mission name from the fourth argument
  local case_name=$5       # Capture the test case name from the fifth argument

  # If the command status is not equal to 0, it indicates failure
  if [ $command_status -ne 0 ]; then
      echo "Test failed on attempt $attempt_i for $mission_name $type_name $case_name."  # Print failure message
      exit 1  # Exit the script
  fi
}

source tests/scripts/_gpu_check.sh

# Path to the YAML configuration file
CONFIG_FILE="tests/scripts/functional_tests/config.yml"

# Function to parse the configuration file and run tests
test_mission() {
  local _type=$1
  local _mission=$2
  # Use parse_config.py to parse the YAML file with test type and test mission
  local _cases=$(python tests/scripts/functional_tests/parse_config.py --config $CONFIG_FILE --type $_type --mission $_mission)

  # Convert the parsed test cases to an array
  IFS=' ' read -r -a _cases <<< "$_cases"

  # Check if _cases is not an empty list
  if [ ${#_cases[@]} -eq 0 ]; then
    echo "No test cases found for mission '$_mission' with test type '$_type'. Exiting."
    exit 0
  fi

  # Loop through each test case, remove leading '-', and run the test
  for _case in "${_cases[@]}"; do
    # Remove leading '-'
    _case=${_case#-}

    test_times=1
    # Attempt to run the test specified number of times
    for attempt_i in $(seq 1 $test_times); do
      echo "---------"
      echo "Attempt $attempt_i for mission ${_mission} with type ${_type} and case: ${_case}"
      echo "---------"

      wait_for_gpu

      # Remove previous results if they exist
      result_path="tests/functional_tests/test_cases/${_type}/${_mission}/results_test/${_case}"
      if [ -d "$result_path" ]; then
        rm -r "$result_path"
      fi

      if [ "${_type}" = "train" ]; then
        run_command "python run.py --config-path tests/functional_tests/test_cases/${_type}/${_mission}/conf --config-name ${_case} action=test" $attempt_i $_mission $_type $_case
        run_command "pytest -p no:warnings -s tests/functional_tests/test_utils/test_equal.py --test_path=tests/functional_tests/test_cases --test_type=${_type} --test_mission=${_mission} --test_case=${_case}" $attempt_i $_mission $_type $_case
      fi

      if [ "${_type}" = "serve" ]; then
        run_command "python run.py --config-path tests/functional_tests/test_cases/${_type}/${_mission}/conf --config-name ${_case} action=run; sleep 1m" $attempt_i $_mission $_type $_case # Serve start
        run_command "python tests/functional_tests/test_cases/${_type}/${_mission}/test_call.py" $attempt_i $_mission $_type $_case # Call
        run_command "python run.py --config-path tests/functional_tests/test_cases/${_type}/${_mission}/conf --config-name ${_case} action=stop" $attempt_i $_mission $_type $_case # Serve stop
      fi

      # Ensure that pytest check is completed before deleting the folder
      sleep 10s
    done
    echo "All $test_times attempts successful for case $_case for mission ${_mission}."
  done
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --type) type="$2"; shift ;;
        --mission) mission="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate that the required parameters --type and --mission are provided
if [ -z "$type" ]; then
  echo "Error: --type is required"
  exit 1
fi

if [ -z "$mission" ]; then
  echo "Error: --mission is required"
  exit 1
fi

# Run the tests based on the provided test type and test mission
test_mission "$type" "$mission"
