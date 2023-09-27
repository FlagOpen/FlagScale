#!/bin/bash

set -u
    HOSTFILE=<xxxx>
set +u
NODES_NUM=$(awk '{$1=$1;print}' $HOSTFILE | wc -l)
echo "NODES_NUM": $NODES_NUM

for ((i=1;i<=$NODES_NUM;i++ )); do
    ip=`sed -n $i,1p $HOSTFILE|cut -f 1 -d" "`
    echo "IP": $ip
    ssh $ip "pkill -f '/usr/bin/python /usr/local/bin/torchrun'" 
done
