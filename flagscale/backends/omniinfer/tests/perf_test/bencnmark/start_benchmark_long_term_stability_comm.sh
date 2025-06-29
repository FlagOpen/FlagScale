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
TASK_OUTPUT_PATH="/tmp/perf_test_output"
LEVEL_INPUT="lts"
FORWARD_TIME=45
FORWARD_TIME_SEND=$(awk -v a=$FORWARD_TIME 'BEGIN{print a/1000}')
MODEL_NAME=deepseek
ADDRESS="127.0.0.1:7000"
MODEL_PATH="/data/models/DeepSeek-R1-w8a8"
PARALLEL_NUM=1536

# 帮助信息
print_help() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  -h, --help                        显示此帮助信息"
    echo "  --forward_time                    FORWARD_TIME (默认: $FORWARD_TIME)"
    echo "  --model_name                      模型名 (默认: $MODEL_NAME)"
    echo "  --model_path                      模型权重路径 (默认: $MODEL_PATH)"
    echo "  --task_output_path                测试结果输出路径 (默认: $TASK_OUTPUT_PATH)"
    echo "  --address                         nginx请求address (默认: $ADDRESS)"
    echo "  --parallel_num                    变长用例并发数 (默认: $PARALLEL_NUM)"
    exit 0
}

# 解析长选项
parse_long_option() {
    case "$1" in
        --task_output_path)
            TASK_OUTPUT_PATH="$2"
            ;;
        --forward_time)
            FORWARD_TIME="$2"
            ;;
        --model_name)
            MODEL_NAME="$2"
            ;;
        --model_path)
            MODEL_PATH="$2"
            ;;
        --address)
            ADDRESS="$2"
            ;;
        --parallel_num)
            PARALLEL_NUM="$2"
            ;;
        --help)
            print_help
            ;;
        *)
            echo "未知选项: $1" >&2
            print_help
            ;;
    esac
    return 0
}

# 解析选项
# 主循环修改后
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            ;;
        --*)
            parse_long_option "$1" "$2"  # 解析但不移位
            shift 2  # 在主循环中统一移位
            ;;
        *)
            echo "未知选项: $1" >&2
            print_help
            ;;
    esac
done

export PYTHONPATH=${PYTHONPATH}:$SCRIPT_PATH/..:$SCRIPT_PATH/../../../tools/llm_evaluation/benchmark_tools

if [ ! -d "$TASK_OUTPUT_PATH" ]; then
    mkdir -p ${TASK_OUTPUT_PATH}
fi

TARGET_URL="http://${ADDRESS}/v1/completions"
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
    RESPONSE_CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "Content-Type: application/json" -d '{"model": "deepseek", "prompt": "Hello, How are you?", "max_tokens": 50}' --connect-timeout 5 --max-time 10 "$TARGET_URL")
    CURL_EXIT=$?

    # 检查请求是否成功
    if [ $CURL_EXIT -eq 0 ] && [ "$RESPONSE_CODE" == "$SUCCESS_CODE" ]; then
        ELAPSED=$((CURRENT_TIME - START_TIME))
        log_info "服务已可用！"
        log_info "最终状态码: $RESPONSE_CODE"
        break
    fi
    sleep 10
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
  --host=${ADDRESS}

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
  --host=${ADDRESS}    \
  --served_model_name=${MODEL_NAME}
}


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

    if [ "$current_hour" -ge 8 ]; then
        log_info "当前时间已超过8:00 ($(date +"%H:%M")，停止循环"
        break
    fi

    # 执行函数
    start_benchmark
    start_variable_benchmark

    # 增加尝试计数
    ATTEMPT=$((ATTEMPT + 1))
    sleep 5  # 每次循环间隔5秒
done

log_info "结果已输出至${TASK_OUTPUT_PATH}目录"

exit 0
