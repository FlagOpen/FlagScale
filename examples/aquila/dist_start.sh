#!/bin/bash
set -u
  FlagScale_HOME=<xxxx>
  PROJ_HOME=<xxxx>
  EXPNAME=<xxxx>
  DATA_PATH=<xxxx>
  HOSTFILE=<xxxx>
  LOG_FILE="examples/aquila/$EXPNAME.log"
  # If trarining 7B, SCRIPT_FILE="examples/aquila/7B/pretrain_aquila_7b_distributed_A800_12n_80g.sh"
  # If trarining 34B, SCRIPT_FILE="examples/aquila/34B/pretrain_aquila_34b_distributed_A100_64n_40g.sh"
  SCRIPT_FILE=<xxxx>
set +u

COUNT=0
hostlist=$(cat $HOSTFILE | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
  echo $host, "bash -c 'cd $FlagScale_HOME; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  ssh -f -n $host "bash -c 'cd $FlagScale_HOME; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  ((COUNT++))
done
