export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export GLOO_SOCKET_IFNAME=enp67s0f5
export TP_SOCKET_IFNAME=enp67s0f5
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=0
export HCCL_OP_EXPANSION_MODE=AIV

# export VLLM_ENABLE_PROFILING=1
# export VLLM_TORCH_PROFILER_DIR=/home/yjf/profiling

export ENABLE_MOE_EP=1
unset ENABLE_ALLTOALL
unset DP_SIZE
unset VLLM_ENABLE_PROFILING
unset VLLM_TORCH_PROFILER_DIR

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m vllm.entrypoints.openai.api_server   --model /opt/models/models/dsv3/DeepSeek-V3-w8a8-0208-50/   --tensor-parallel-size 16  --gpu-memory-utilization 0.8   --dtype bfloat16    --block-size 128   --trust-remote-code  --served-model-name deepseek --distributed-executor-backend=ray --max-model-len=1024 --host="127.0.0.1" --port=8999

# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export GLOO_SOCKET_IFNAME=enp67s0f5
# export TP_SOCKET_IFNAME=enp67s0f5
# # export GLOO_SOCKET_IFNAME=eth4
# # export TP_SOCKET_IFNAME=eth4
# export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
# export RAY_DEDUP_LOGS=0
# export HCCL_OP_EXPANSION_MODE="AIV"

# export ENABLE_MOE_EP=1
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# VLLM_TORCH_PROFILER_DIR=/home/yjf/profiling python -m vllm.entrypoints.openai.api_server --model /home/yjf/DeepSeek-V3-w8a8-0208-50 --tensor-parallel-size 16  --gpu-memory-utilization 1.0  --dtype bfloat16 --block-size 128 --trust-remote-code --served-model-name deepseek --distributed-executor-backend=ray --max-model-len=1024  --host="127.0.0.1"  --port=8999
