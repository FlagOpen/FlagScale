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
source "$SCRIPT_PATH/../../../log_utils.sh"
OUTPUT_DIR=${OUTPUT_DIR-"output"}
OUTPUT_PATH=${OUTPUT_PATH-"/home/ma-user/modelarts/outputs/train_url_0"}
PD_SEPARATION_PORT=${PD_SEPARATION_PORT-"8001"}
TASK_OUTPUT_PATH=${OUTPUT_PATH}/${OUTPUT_DIR}
LEVEL_INPUT=level1
FORWARD_TIME_SEND=0.01
MODEL_NAME=deepseek

export PYTHONPATH=${PYTHONPATH}:$SCRIPT_PATH/..:$SCRIPT_PATH/../../../tools/llm_evaluation/benchmark_tools

if [ ! -d "$TASK_OUTPUT_PATH" ]; then
    mkdir -p ${TASK_OUTPUT_PATH}
fi
ENV_PATH=$SCRIPT_PATH/../../../env.sh
MOCK_MODEL_ENV_PATH=$SCRIPT_PATH/../mock_model_env.sh
source ${MOCK_MODEL_ENV_PATH}
source ${ENV_PATH}

HOST="127.0.0.1:${PD_SEPARATION_PORT}"
TARGET_URL="http://${HOST}/v1/completions"
TIMEOUT=300
SUCCESS_CODE=200


START_TIME=$(date +%s)
END_TIME=$((START_TIME + TIMEOUT))

log_info "检测服务状态"
while true; do
    CURRENT_TIME=$(date +%s)

    # 检查是否超时
    if [ $CURRENT_TIME -ge $END_TIME ]; then
        log_error "超过 ${TIMEOUT} 秒未检测到服务可用"
        exit 1
    fi

    # 发送请求并获取状态码
    RESPONSE_CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "Content-Type: application/json" -d '{"model": "deepseek", "prompt": "Hello, How are you?", "max_tokens": 50}' --connect-timeout 5 --max-time 10 "$TARGET_URL") || true

    # 检查请求是否成功
    if [ "$RESPONSE_CODE" == "$SUCCESS_CODE" ]; then
        ELAPSED=$((CURRENT_TIME - START_TIME))
        log_info "服务已可用！"
        log_info "最终状态码: $RESPONSE_CODE"
        break
    fi
    sleep 10
done


function start_benchmark {
  log_info "FORWARD_TIME 的值: ${FORWARD_TIME}"
  log_info "PARALLEL_NUM 的值: ${PARALLEL_NUM}"
  log_info "TASK_OUTPUT_PATH 的值: ${TASK_OUTPUT_PATH}"

  log_info "执行benchmark用例"
  cd $SCRIPT_PATH/

  pytest -vsra --disable-warnings -m "platform_ascend910b and ${LEVEL_INPUT}" \
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
export RANDOM_MODE=1
export KV_CACHE_MOD=1
export FORWARD_TIME=15
FORWARD_TIME_SEND=0.015
start_benchmark

exit 0

