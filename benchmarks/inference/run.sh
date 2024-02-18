#!/bin/bash

# download dataset
# cd $FlagScale_HOME/benchmarks/inference/
# wget -O test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl

MODEL_INFO=Aquila-1.8B
CHECKPOINT=<xxxx>
MASTER_PORT=8888
DEVICES=0
TYPE=throughout # throughout/latency/serving

FlagScale_HOME=<xxxx>
EXPNAME=$TYPE-test
LOG_FILE=$FlagScale_HOME/benchmarks/inference/log.$EXPNAME
SCRIPT_FILE=$FlagScale_HOME/benchmarks/inference/benchmark.sh

cd $FlagScale_HOME/benchmarks/inference
nohup bash $SCRIPT_FILE $MODEL_INFO $CHECKPOINT $MASTER_PORT $DEVICES $TYPE > $LOG_FILE 2>&1 &
