#!/bin/bash

source /root/miniconda3/bin/activate flagscale-inference
mkdir -p /home/FlagScale/tests/benchmarks/test_cases/inference/deepseek_r1_distill_qwen/results_test/7b-tp2/inference_logs
mkdir -p /home/FlagScale/tests/benchmarks/test_cases/inference/deepseek_r1_distill_qwen/results_test/7b-tp2/inference_logs/pids

cd /home/FlagScale

export PYTHONPATH=/home/FlagScale:${PYTHONPATH}

cmd="HYDRA_FULL_ERROR=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_BENCHMARK=false CUDNN_DETERMINISTIC=true NVTE_APPLY_QK_LAYER_SCALING=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_ALGO=Ring NCCL_PROTOCOL=LLC SEED=1234 PYTHONHASHSEED=0 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SCIPY_RDRANDOM=0 TF_DETERMINISTIC_OPS=1 TORCH_CUDNN_DETERMINISM=True CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=INFO MAGIC_CACHE=disabled python flagscale/benchmarks/benchmark_throughput.py --config-path=/home/FlagScale/tests/benchmarks/test_cases/inference/deepseek_r1_distill_qwen/results_test/7b-tp2/inference_logs/scripts/inference.yaml"

bash -c "$cmd; sync"  >> /home/FlagScale/tests/benchmarks/test_cases/inference/deepseek_r1_distill_qwen/results_test/7b-tp2/inference_logs/host_0_localhost.output 

