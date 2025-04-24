#!/bin/bash
set -xe
export NCCL_SOCKET_IFNAME=bond0 && export FLAGCX_SOCKET_IFNAME=bond0 && export VLLM_USE_V1=0 && export FLAGCX_PATH=/mine/ip122/tune_qwen/FlagCX/ && export FLAGCX_DEBUG=TRACE && export FLAGCX_DEBUG_SUBSYS=ALL && export NCCL_DEBUG=TRACE && export NCCL_DEBUG_SUBSYS=ALL
# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

python3 disagg_prefill_proxy_xpyd.py &

#MODEL_NAME=${HF_MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}

MODEL_NAME=/models/Qwen2.5-0.5B-Instruct

## 2P2D, TP=1
## prefilling instance, which is the KV producer
#CUDA_VISIBLE_DEVICES=4 vllm serve $MODEL_NAME \
#    --host 0.0.0.0 \
#    --port 20001 \
#    --served-model-name base_model \
#    --max-model-len 8192 \
#    --gpu-memory-utilization 0.8 \
#    --kv-transfer-config \
#    '{"kv_connector":"P2pConnector","kv_role":"kv_producer","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20001"}}' &
#
## prefilling instance, which is the KV producer
#CUDA_VISIBLE_DEVICES=5 vllm serve $MODEL_NAME \
#    --host 0.0.0.0 \
#    --port 20002 \
#    --served-model-name base_model \
#    --max-model-len 8192 \
#    --gpu-memory-utilization 0.8 \
#    --kv-transfer-config \
#    '{"kv_connector":"P2pConnector","kv_role":"kv_producer","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20002"}}' &
#
## decoding instance, which is the KV consumer
#CUDA_VISIBLE_DEVICES=6 vllm serve $MODEL_NAME \
#    --host 0.0.0.0 \
#    --port 20003 \
#    --served-model-name base_model \
#    --max-model-len 8192 \
#    --gpu-memory-utilization 0.8 \
#    --kv-transfer-config \
#    '{"kv_connector":"P2pConnector","kv_role":"kv_consumer","kv_port":"23001","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20003"}}' &
#
## decoding instance, which is the KV consumer
#CUDA_VISIBLE_DEVICES=7 vllm serve $MODEL_NAME \
#    --host 0.0.0.0 \
#    --port 20004 \
#    --served-model-name base_model \
#    --max-model-len 8192 \
#    --gpu-memory-utilization 0.8 \
#    --kv-transfer-config \
#    '{"kv_connector":"P2pConnector","kv_role":"kv_consumer","kv_port":"24001","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20004"}}' &


# 2P2D, TP=2
# prefilling instance, which is the KV producer
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_NAME \
    --host 0.0.0.0 \
    --port 20001 \
    --tensor-parallel-size 1 \
    --served-model-name base_model \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"P2pConnector","kv_role":"kv_producer","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20001"}}' &

# decoding instance, which is the KV consumer
CUDA_VISIBLE_DEVICES=2 vllm serve $MODEL_NAME \
    --host 0.0.0.0 \
    --port 20003 \
    --tensor-parallel-size 1 \
    --served-model-name base_model \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config \
    '{"kv_connector":"P2pConnector","kv_role":"kv_consumer","kv_port":"23001","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20003"}}' &
