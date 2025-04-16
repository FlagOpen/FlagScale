#!/bin/bash

set -x

source /root/miniconda3/bin/activate flagscale-inference && export GLOO_SOCKET_IFNAME=bond0

if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=/root/miniconda3/envs/flagscale-inference/lib/python3.12/site-packages:/mine/ip122/tune_qwen/github_flagscale
else
    export PYTHONPATH="$PYTHONPATH:/root/miniconda3/envs/flagscale-inference/lib/python3.12/site-packages:/mine/ip122/tune_qwen/github_flagscale"
fi

ray_path=$(realpath $(which ray))
# clean nodes 
ssh -n -p 22 10.1.1.108 "docker exec ds /bin/bash -c 'source /root/miniconda3/bin/activate flagscale-inference && export GLOO_SOCKET_IFNAME=bond0 && ${ray_path} stop'"
source /root/miniconda3/bin/activate flagscale-inference && export GLOO_SOCKET_IFNAME=bond0 && ${ray_path} stop
pkill -f 'run_inference_engine'
pkill -f 'run_fs_serve_vllm'
pkill -f 'vllm serve'

# start cluster
# master node
source /root/miniconda3/bin/activate flagscale-inference && export GLOO_SOCKET_IFNAME=bond0 && ${ray_path} start --head --port=59081 --num-gpus=8

# worker nodes
ssh -n -p 22 10.1.1.108 "docker exec ds /bin/bash -c 'source /root/miniconda3/bin/activate flagscale-inference && export GLOO_SOCKET_IFNAME=bond0 && ${ray_path} start --address=10.1.1.122:59081 --num-gpus=8'"
mkdir -p /mine/ip122/tune_qwen/github_flagscale/outputs/deepseek_v3/serve_logs
mkdir -p /mine/ip122/tune_qwen/github_flagscale/outputs/deepseek_v3/serve_logs/pids

cd /mine/ip122/tune_qwen/github_flagscale

cmd="CUDA_DEVICE_MAX_CONNECTIONS=1 python flagscale/serve/run_inference_engine.py --config-path=/mine/ip122/tune_qwen/github_flagscale/outputs/deepseek_v3/serve_logs/scripts/serve.yaml --log-dir=/mine/ip122/tune_qwen/github_flagscale/outputs/deepseek_v3/serve_logs"

nohup bash -c "$cmd; sync" >> /mine/ip122/tune_qwen/github_flagscale/outputs/deepseek_v3/serve_logs/host_0_localhost.output 2>&1 & echo $! > /mine/ip122/tune_qwen/github_flagscale/outputs/deepseek_v3/serve_logs/pids/host_0_localhost.pid