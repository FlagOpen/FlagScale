#!/bin/bash

run_command() {
  eval $1
  if [ $? -ne 0 ]; then
    echo "Command failed: $1"
    exit 1
  fi
}

run_train() {
  local config_path=$1
  run_command "python run.py --config-path $config_path --config-name config action=test"
}

run_pytest() {
  local results_path=$1
  run_command "pytest -p no:warnings -s tests/functional_tests/test_result.py --test_reaults_path=$results_path"
}

run_train "tests/functional_tests/aquila/conf"
run_pytest "./tests/functional_tests/aquila/test_result"

run_train "tests/functional_tests/mixtral/conf"
run_pytest "./tests/functional_tests/mixtral/test_result"