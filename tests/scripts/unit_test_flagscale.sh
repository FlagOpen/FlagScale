#!/bin/bash

if [ -z "$1" ]
then
  code_id=0
else
  code_id=$1
fi

run_pytest() {
  local test_path=$1
  torchrun --nproc_per_node=8 -m pytest --import-mode=importlib --cov-append --cov-report=html:/workspace/report/$code_id/cov-report-flagscale --cov=flagscale -q -x -p no:warnings $test_path
  if [ $? -ne 0 ]; then
    echo "Pytest failed for $test_path"
    exit 1
  fi
}

run_pytest tests/unit_tests/launcher
