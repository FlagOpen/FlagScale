#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -e

# 获取当前脚本的绝对路径并进入
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_PATH

# 导入日志工具函数
source "$SCRIPT_PATH/../../log_utils.sh"
OUTPUT_DIR=${OUTPUT_DIR-"output"}
TASK_OUTPUT_PATH=${OUTPUT_PATH}/${OUTPUT_DIR}
LEVEL_INPUT=level0
FORWARD_TIME_SEND=0.015
MODEL_NAME=deepseek
HOST="127.0.0.1:8000"
MODEL_PATH="/home/ma-user/modelarts/inputs/data_url_0/model/DeepSeek-V2-Lite"
FLAG=$1

export PYTHONPATH=${PYTHONPATH}:$SCRIPT_PATH/..:$SCRIPT_PATH/../../../tools/llm_evaluation/benchmark_tools

if [ ! -d "$TASK_OUTPUT_PATH" ]; then
    mkdir -p ${TASK_OUTPUT_PATH}
fi

function start_benchmark {
  log_info "vllm服务化启动"
  touch ${TASK_OUTPUT_PATH}/${LOGFILE}
  log_info "指定模型路径: ${MODEL_PATH}"
  log_info "TASK_OUTPUT_PATH 的值: ${TASK_OUTPUT_PATH}"
  log_info "FORWARD_TIME 的值: ${FORWARD_TIME}"
  log_info "PARALLEL_NUM 的值: ${PARALLEL_NUM}"
  log_info "PROMPT_TOKENS 的值: ${PROMPT_TOKENS}"
  log_info "OUTPUT_TOKENS 的值: ${OUTPUT_TOKENS}"
  log_info "BENCHMARK_CSV 的值: ${BENCHMARK_CSV}"
  nohup bash $SCRIPT_PATH/../../start_vllm_server/deepseek_dp8_tp1_ep1.sh ${MODEL_PATH}/ &> ${TASK_OUTPUT_PATH}/${LOGFILE} &

  log_info "进入循环等待服务启动标识符出现"
  cost=0
  interval=10
  endtime=300
  while true; do
      if [ $cost -gt $endtime ]; then
          log_error "等待服务启动时间超过${endtime}秒，退出循环"
          log_error "服务启动日志结尾部分内容:"
          tail -n 50 ${TASK_OUTPUT_PATH}/${LOGFILE}
          break
      fi
      if grep -q "Application startup complete" ${TASK_OUTPUT_PATH}/${LOGFILE}; then
          log_info "服务已启动，继续执行用例"
          log_info "服务启动日志开头部分内容:"
          head -n 50 ${TASK_OUTPUT_PATH}/${LOGFILE}
          break
      else
          log_warning "服务启动中，等待${interval}秒"
          sleep ${interval}
          cost=$((cost + interval))
      fi
  done

  log_info "执行benchmark用例"
  cd $SCRIPT_PATH/../../
  log_info "当前目录: $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  pytest -vsra --disable-warnings -m "platform_ascend910b and common" \
  ./perf_test/bencnmark/test_mock_model_benchmark.py \
  --output_dir=${TASK_OUTPUT_PATH} \
  --forward_time=${FORWARD_TIME_SEND} \
  --except_time=0.003 \
  --tokenizer=${MODEL_PATH} \
  --served_model_name=${MODEL_NAME} \
  --host=${HOST} \
  --parallel_num="${PARALLEL_NUM}" \
  --prompt_tokens="${PROMPT_TOKENS}" \
  --output_tokens="${OUTPUT_TOKENS}" \
  --benchmark_csv="${BENCHMARK_CSV}"

  log_info "停止vllm服务"
  pkill -f python3
}

# 根据参数执行不同的步骤
case "$1" in
  2p1d_3)
    log_info "开始执行2p1d，3机性能测试用例"
    PARALLEL_NUM="[80]"
    PROMPT_TOKENS="[2048, 2048, 3072, 3500]"
    OUTPUT_TOKENS="[1024, 2048, 1024, 1024]"
    BENCHMARK_CSV="benchmark_result_2p1d_3.csv"
    ;;
  2p1d_4)
    log_info "开始执行2p1d，4机性能测试用例"
    PARALLEL_NUM="[160, 1536, 2304]"
    PROMPT_TOKENS="[2048, 2048, 3072, 3500, 6000]"
    OUTPUT_TOKENS="[1024, 2048, 1024, 1024, 1000]"
    BENCHMARK_CSV="benchmark_result_2p1d_4.csv"
    ;;
  *)
    log_info "无效的参数，请使用 2p1d_3 或 2p1d_4"
    exit 1
    ;;
esac

log_info "开始执行FORWARD_TIME=15ms性能测试用例"
export RANDOM_MODE=1
export KV_CACHE_MOD=1
export FORWARD_TIME=15
FORWARD_TIME_SEND=0.015
LOGFILE="server_${FLAG}_15.log"
BENCHMARK_CSV="benchmark_result_${1}_15ms.csv"
start_benchmark
log_info "开始执行FORWARD_TIME=30ms性能测试用例"
export RANDOM_MODE=1
export KV_CACHE_MOD=1
export FORWARD_TIME=30
FORWARD_TIME_SEND=0.030
LOGFILE="server_${FLAG}_30.log"
BENCHMARK_CSV="benchmark_result_${1}_30ms.csv"
start_benchmark
log_info "开始执行FORWARD_TIME=45ms性能测试用例"
export RANDOM_MODE=1
export KV_CACHE_MOD=1
export FORWARD_TIME=45
FORWARD_TIME_SEND=0.045
LOGFILE="server_${FLAG}_45.log"
BENCHMARK_CSV="benchmark_result_${1}_45ms.csv"
start_benchmark

IP=$(ip route get 1.2.3.4 | awk '{print $7}' | head -1)

log_info "结果已输出至${IP}下的/workspace/CI/outputs/output目录"

# 定义清理函数
cleanup() {
    log_info "停止vllm服务"
    pkill -f python3
    exit $?
}

# 捕获退出信号并执行清理函数
trap cleanup EXIT

exit 0

