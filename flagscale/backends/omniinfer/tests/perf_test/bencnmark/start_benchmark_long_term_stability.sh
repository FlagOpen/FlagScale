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
LEVEL_INPUT="lts"
FORWARD_TIME_SEND=0.045
MODEL_NAME=deepseek
HOST="127.0.0.1:8000"
MODEL_PATH="/home/ma-user/modelarts/inputs/data_url_0/model/DeepSeek-V2-Lite"

export PYTHONPATH=${PYTHONPATH}:$SCRIPT_PATH/..:$SCRIPT_PATH/../../../tools/llm_evaluation/benchmark_tools

if [ ! -d "$TASK_OUTPUT_PATH" ]; then
    mkdir -p ${TASK_OUTPUT_PATH}
fi

export KV_CACHE_MOD=1
export RANDOM_MODE=1
export FORWARD_TIME=45

log_info "vllm服务化启动"
touch ${TASK_OUTPUT_PATH}/server_lts.log
log_info "指定模型路径: ${MODEL_PATH}"
log_info "FORWARD_TIME 的值: ${FORWARD_TIME}"
nohup bash $SCRIPT_PATH/../../start_vllm_server/deepseek_dp8_tp1_ep1.sh ${MODEL_PATH}/ &> ${TASK_OUTPUT_PATH}/server_lts.log &

log_info "进入循环等待服务启动标识符出现"
cost=0
interval=20
endtime=300
while true; do
    if [ $cost -gt $endtime ]; then
        log_error "等待服务启动时间超过${endtime}秒，退出循环"
        log_error "服务启动日志结尾部分内容:"
        tail -n 50 ${TASK_OUTPUT_PATH}/server_lts.log
        break
    fi
    if grep -q "Application startup complete" ${TASK_OUTPUT_PATH}/server_lts.log; then
        log_info "服务已启动，继续执行用例"
        log_info "服务启动日志开头部分内容:"
        head -n 50 ${TASK_OUTPUT_PATH}/server_lts.log
        break
    else
        log_warning "服务启动中，等待${interval}秒"
        sleep ${interval}
        cost=$((cost + interval))
    fi
done

function start_benchmark {

  log_info "开始执行daily的FORWARD_TIME=45ms性能测试用例"
  cd $SCRIPT_PATH/../../
  log_info "当前目录: $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  pytest -vsra --disable-warnings -m "platform_ascend910b and ${LEVEL_INPUT}" \
  ./perf_test/bencnmark/test_mock_model_benchmark.py \
  --output_dir=${TASK_OUTPUT_PATH} \
  --forward_time=${FORWARD_TIME_SEND} \
  --except_time=0.003 \
  --tokenizer=${MODEL_PATH} \
  --served_model_name=${MODEL_NAME} \
  --host=${HOST}
}


function start_variable_benchmark {
  log_info "开始执行daily的FORWARD_TIME=${FORWARD_TIME}性能测试用例"
  cd $SCRIPT_PATH/variablelength/
  log_info "当前目录: $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  pytest -vsra --disable-warnings -m "platform_ascend910b and level1" \
  ./test_mock_model_variable_length.py \
  --output_dir=${TASK_OUTPUT_PATH} \
  --except_time=0.003 \
  --forward_time=${FORWARD_TIME_SEND} \
  --parallel_num="${PARALLEL_NUM}" \
  --host=${HOST}    \
  --served_model_name=${MODEL_NAME}
}

if [ -z "$1" ]; then
    PARALLEL_NUM=1536
else
    PARALLEL_NUM=$1
fi

MAX_ATTEMPTS=20  # 最大尝试次数
ATTEMPT=1        # 当前尝试次数

while true; do
    # 检查是否超过最大尝试次数
    if [ $ATTEMPT -gt $MAX_ATTEMPTS ]; then
        log_info "已达到最大尝试次数 $MAX_ATTEMPTS，停止循环"
        break
    fi

    # 检查当前时间是否超过8:00 (24小时制)
    current_hour=$(date +%H)
    current_minute=$(date +%M)

    if [ "$current_hour" -ge 8 ] && [ "$current_hour" -le 21 ]; then
        log_info "当前时间在8:00 ($(date +"%H:%M") 到 21:00之间，停止循环"
        break
    fi

    # 执行函数
    start_benchmark
    start_variable_benchmark

    # 增加尝试计数
    ATTEMPT=$((ATTEMPT + 1))
    sleep 5  # 每次循环间隔5秒
done

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

