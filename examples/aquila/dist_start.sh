#!/bin/bash
set -u
  FlagScale_HOME=<xxxx>
  PROJ_HOME=<xxxx>
  EXPNAME=<xxxx>
  DATA_PATH=<xxxx>
  HOSTFILE=<xxxx>
  LOG_FILE="../examples/aquila/$EXPNAME.log"
  SCRIPT_FILE=<xxxx>
set +u

COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
  echo $host, "bash -c 'cd $FlagScale_HOME; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  ssh -f -n $host "bash -c 'cd $FlagScale_HOME; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" >> $LOG_FILE.$COUNT.$host 2>&1 &'"
  ((COUNT++))
done
