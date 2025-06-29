#!/bin/bash

CRTDIR=$(pwd)
host=$1
case_level=$2
export PYTHONPATH=${CRTDIR}:$PYTHONPATH

python simple_evals.py --dataset mgsm drop mmlu humaneval gpqa \
--served-model-name deepseek \
--url http://${host}/v1 \
--max-tokens 2048 \
--temperature 0.5 \
--num-threads 500 \
--debug
