#!/bin/bash

# Extract information from the hostfile
echo "HOSTFILE: $HOSTFILE"
NUM_NODES=$(awk 'BEGIN {cnt=0} !/^#/ && NF {$1=$1; cnt++} END {print cnt}' "$HOSTFILE")
echo "NUM_NODES: $NUM_NODES"
NODE_ADDR=$(ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|head -n 1)
NODE_NAME=$(hostname)
NODE_RANK_BY_ADDR=$(awk -v node="$NODE_ADDR" 'BEGIN {cnt=0} !/^#/ && NF {ranks[$1]=cnt; cnt++;} END {print ranks[node];}' "$HOSTFILE")
echo "NODE_RANK_BY_ADDR: $NODE_RANK_BY_ADDR"
NODE_RANK_BY_NAME=$(awk -v node="$NODE_NAME" 'BEGIN {cnt=0} !/^#/ && NF {ranks[$1]=cnt; cnt++;} END { print ranks[node];}' "$HOSTFILE")
echo "NODE_RANK_BY_NAME: $NODE_RANK_BY_NAME"
if [ -n "$NODE_RANK_BY_ADDR" ]; then
    NODE_RANK=$NODE_RANK_BY_ADDR
    echo "NODE_RANK: $NODE_RANK"
    NODE_DEVICES=$(awk -v node="$NODE_ADDR" '!/^#/ && NF && $1==node {split($2, arr, "="); print arr[2]}' "$HOSTFILE")
    echo "NODE_DEVICES: $NODE_DEVICES"
    NODE_TYPE=$(awk -v node="$NODE_ADDR" '!/^#/ && NF && $1==node {print $3}' "$HOSTFILE")
    echo "NODE_TYPE: $NODE_TYPE"
elif [ -n "$NODE_RANK_BY_NAME" ]; then
    NODE_RANK=$NODE_RANK_BY_NAME
    echo "NODE_RANK: $NODE_RANK"
    NODE_DEVICES=$(awk -v node="$NODE_NAME" '!/^#/ && NF && $1==node {split($2, arr, "="); print arr[2]}' "$HOSTFILE")
    echo "NODE_DEVICES: $NODE_DEVICES"
    NODE_TYPE=$(awk -v node="$NODE_NAME" '!/^#/ && NF && $1==node {print $3}' "$HOSTFILE")
    echo "NODE_TYPE: $NODE_TYPE"
else
    echo "Error: NODE_RANK not found"
    exit 1
fi
MASTER_ADDR=$(grep -v '^#\|^$' $HOSTFILE | head -n1 | awk '{print $1;}')
echo "MASTER_ADDR: $MASTER_ADDR"
MASTER_PORT=12346
echo "MASTER_PORT: $MASTER_PORT"

# Export the environment variables
# NOTE: Please change the following envrioment variables base on the cluster configuration
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_IB_TIMEOUT=12
# export NCCL_IB_RETRY_CNT=7
# export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4
export GLOO_SOCKET_IFNAME=eth0
if [ "$NODE_TYPE" == "A100" ]; then
  export NCCL_IB_HCA=mlx5_2,mlx5_5
else
  export NCCL_IB_HCA=mlx5_2,mlx5_8
fi
echo "NCCL_IB_HCA, $NCCL_IB_HCA"
