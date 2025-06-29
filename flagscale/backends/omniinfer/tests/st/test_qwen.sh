#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
set -e

# 获取当前脚本的绝对路径并进入
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 导入日志工具函数
source "$SCRIPT_PATH/../log_utils.sh"

TASK_OUTPUT_PATH="/data/omni_infer/log/"
MODEL_NAME="qwen"
MODEL_PATH="/data/models/Qwen2.5-0.5B-Instruct"
HOST="127.0.0.1:8300"
LEVEL_INPUT="level0"

# 解析长选项
parse_long_option() {
    case "$1" in
        --host)
            HOST="$2"
            ;;
        --task-output-path)
            TASK_OUTPUT_PATH="$2"
            ;;
        --model-path)
            MODEL_PATH="$2"
            ;;
    esac
    return 0
}

# 解析选项
while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            parse_long_option "$1" "$2"
            shift 2
            ;;
        *)
            echo "未知选项: $1" >&2
            print_help
            ;;
    esac
done

if [ ! -d "$TASK_OUTPUT_PATH" ]; then
    mkdir -p ${TASK_OUTPUT_PATH}
fi

# 先终止环境上其他模型启动的服务
pkill -f python || true

log_info "vllm服务化启动"
touch ${TASK_OUTPUT_PATH}/server_qwen.log
nohup bash $SCRIPT_PATH/../start_vllm_server/qwen_dp1_tp1.sh ${MODEL_PATH}/ > ${TASK_OUTPUT_PATH}/server_qwen.log 2>&1 &

log_info "进入循环等待服务启动标识符出现"
cost=0
interval=10
endtime=300
while true; do
    if [ $cost -gt $endtime ]; then
        log_error "等待服务启动时间超过${endtime}秒，退出循环"
        log_error "服务启动日志结尾部分内容:"
        tail -n 50 ${TASK_OUTPUT_PATH}/server_qwen.log
        break
    fi
    if grep -q "Application startup complete" ${TASK_OUTPUT_PATH}/server_qwen.log; then
        log_info "服务已启动，继续执行用例"
        log_info "服务启动日志开头部分内容:"
        head -n 50 ${TASK_OUTPUT_PATH}/server_qwen.log
        break
    else
        log_warning "服务启动中，等待${interval}秒"
        sleep ${interval}
        cost=$((cost + interval))
    fi
done

log_info "执行测试用例"
cd $SCRIPT_PATH
log_info "当前目录: $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pytest -vsra --disable-warnings -m "platform_ascend910b and ${LEVEL_INPUT}" \
  ./precision/test_r1_precision.py --host="127.0.0.1:8300" --served_model_name="qwen"

log_info "停止vllm服务"
pkill -f python


test_result=$?
# 检查退出状态码，如果不为0则表示有测试失败
if [ $test_result -ne 0 ]; then
    log_error "测试失败"
    exit 1
else
    echo "测试例100%通过"
fi

exit 0




