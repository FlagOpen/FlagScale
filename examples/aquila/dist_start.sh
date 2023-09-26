#!/bin/bash
FlagScale_HOME=<xxxx>
PROJ_HOME=<xxxx>
EXPNAME=<xxxx>
DATA_PATH=<xxxx>

set -u
  HOSTFILE=$PROJ_HOME/config/hostfile
  LOG_FILE="examples/aquila/$EXPNAME.log"
  SCRIPT_FILE="examples/aquila/33B/pretrain_aquila_33b_distributed_A100_64n_40g.sh"
set +u

COUNT=0
hostlist=$(cat $HOSTFILE | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
  echo $host, "bash -c 'cd $FlagScale_HOME; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  ssh -f -n $host "bash -c 'cd $FlagScale_HOME; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  ((COUNT++))
done
