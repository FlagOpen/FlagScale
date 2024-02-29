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

command=$1

COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)

# bash run.sh start
if [[ $command == "start" ]]; then
    for host in ${hostlist[@]}; do
      echo $host, "bash -c 'cd $FlagScale_HOME/megatron; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" > $LOG_FILE.$COUNT.$host 2>&1 & echo \$! > $LOG_FILE.$COUNT.$host.pid'"
      ssh -f -n $host "bash -c 'cd $FlagScale_HOME/megatron; nohup bash $SCRIPT_FILE $PROJ_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" > $LOG_FILE.$COUNT.$host 2>&1 & echo \$! > $LOG_FILE.$COUNT.$host.pid'"
      ((COUNT++))
    done
# bash run.sh stop
elif [[ $command == "stop" ]]; then
    for host in ${hostlist[@]}; do
        pid=$(ssh $host "cat $LOG_FILE.$COUNT.$host.pid")
        ssh $host "pkill -P $pid" 
        echo "Process on $host is killed."
        ((COUNT++))
    done
else
    echo "Invalid command. Use 'start' or 'stop'."
    exit 1
fi
