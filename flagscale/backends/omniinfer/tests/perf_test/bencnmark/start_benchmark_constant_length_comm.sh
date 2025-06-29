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
FORWARD_TIME=15
FORWARD_TIME_SEND=$(awk -v a=$FORWARD_TIME 'BEGIN{print a/1000}')
MODEL_NAME=deepseek
ADDRESS="127.0.0.1:7000"
MODEL_PATH="/data/models/DeepSeek-R1-w8a8"
DEVICE="2p1d_4"

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
    echo "  --device                          设备情况 (默认: $DEVICE)"
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
        --device)
            DEVICE="$2"
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
    log_info "等待服务启动。。。"
done


export PYTHONPATH=${PYTHONPATH}:$SCRIPT_PATH/..:$SCRIPT_PATH/../../../tools/llm_evaluation/benchmark_tools

if [ ! -d "$TASK_OUTPUT_PATH" ]; then
    mkdir -p ${TASK_OUTPUT_PATH}
fi

function start_benchmark {
  log_info "指定模型路径: ${MODEL_PATH}"
  log_info "TASK_OUTPUT_PATH 的值: ${TASK_OUTPUT_PATH}"
  log_info "FORWARD_TIME 的值: ${FORWARD_TIME}"
  log_info "PARALLEL_NUM 的值: ${PARALLEL_NUM}"
  log_info "PROMPT_TOKENS 的值: ${PROMPT_TOKENS}"
  log_info "OUTPUT_TOKENS 的值: ${OUTPUT_TOKENS}"
  log_info "BENCHMARK_CSV 的值: ${BENCHMARK_CSV}"
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
  --host=${ADDRESS} \
  --parallel_num="${PARALLEL_NUM}" \
  --prompt_tokens="${PROMPT_TOKENS}" \
  --output_tokens="${OUTPUT_TOKENS}" \
  --benchmark_csv="${BENCHMARK_CSV}"

}

# 根据参数执行不同的步骤
case "$DEVICE" in
  2p1d_3)
    log_info "开始执行2p1d，3机性能测试用例"
    PARALLEL_NUM="[768]"
    PROMPT_TOKENS="[2048, 2048, 3072, 3500]"
    OUTPUT_TOKENS="[1024, 2048, 1024, 1024]"
    BENCHMARK_CSV="benchmark_result_2p1d_3.csv"
    ;;
  2p1d_4)
    log_info "开始执行2p1d，4机性能测试用例"
    PARALLEL_NUM="[768]"
    PROMPT_TOKENS="[2048, 2048, 3072, 3500]"
    OUTPUT_TOKENS="[1024, 2048, 1024, 1024]"
    BENCHMARK_CSV="benchmark_result_2p1d_4.csv"
    ;;
  *)
    log_info "无效的DEVICE参数，请使用 2p1d_3 或 2p1d_4"
    exit 1
    ;;
esac

log_info "开始执行FORWARD_TIME=${FORWARD_TIME}ms性能测试用例"
BENCHMARK_CSV="benchmark_result_${DEVICE}_${FORWARD_TIME}ms.csv"
start_benchmark

log_info "结果已输出至${TASK_OUTPUT_PATH}目录"


exit 0

