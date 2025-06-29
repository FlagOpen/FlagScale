#!/bin/bash

CRTDIR=$(pwd)
host=$1
case_level=$2
# echo "pwd is ${CRTDIR}"
export PYTHONPATH=${CRTDIR}:$PYTHONPATH
# echo "PYTHONPATH is ${PYTHONPATH}"

python main.py --url=http://${host}/v1/chat/completions --parallel_num=10 --max_fail=1000 --model_name=deepseek --case_type=deepseek_v3 --case_time=fast --case_level=${case_level}
# python main.py --url=http://127.0.0.1:8000/v1/chat/completions --parallel_num=10 --max_fail=1000 --model_name=deepseek --case_type=deepseek-v3 --case_time=slow
