#!/bin/bash
MODEL_INFO=Aquila-1.8B
CHECKPOINT=<xxxx>
MASTER_PORT=8888
DEVICES=0
TYPE=throughout

FlagScale_HOME=<xxxx>
EXPNAME=$TYPE-test
LOG_FILE=<xxxx>/$EXPNAME.log
SCRIPT_FILE=benchmark.sh

cd $FlagScale_HOME/benchmarks
nohup bash $SCRIPT_FILE $MODEL_INFO $CHECKPOINT $MASTER_PORT $DEVICES $TYPE > $LOG_FILE 2>&1 &
