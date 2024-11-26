#!/bin/bash

set -u
    HOSTFILE=$1
set +u
NUM_NODES=$(grep -v '^#\|^$' $HOSTFILE | wc -l)
echo "NUM_NODES: $NUM_NODES"

hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
    ssh $host "pkill -f '/usr/local/bin/torchrun'" 
    echo "$host is killed."
done
