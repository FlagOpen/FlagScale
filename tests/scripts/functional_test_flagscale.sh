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
  local test_path=$1
  run_command "pytest -p no:warnings -s tests/functional_tests/test_result.py --test_path=$test_path"
}

run_train "tests/functional_tests/aquila/conf"
run_pytest "./tests/functional_tests/aquila"

run_train "tests/functional_tests/mixtral/conf"
run_pytest "./tests/functional_tests/mixtral"