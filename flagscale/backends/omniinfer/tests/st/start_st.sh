#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

set -e

# 获取当前脚本的绝对路径并进入
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 导入日志工具函数
source "$SCRIPT_PATH/../log_utils.sh"
# 帮助信息
print_help() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  -h, --help                        显示此帮助信息"
    echo "  --host                            vllm服务ip和port"
    echo "  --html                            pytest html报告路径"
    echo "  --task-output-path                任务输出路径"
    echo "  --pd-separation-flag              PD分离标志位，0表示不分离，1表示分离"
    echo "  --model-path                      模型路径"
    echo "  --vllm-env                        VLLM服务化环境变量文件"
    echo "  --mockmodel-env                   Mock model环境变量文件"
    echo "  --test-level                      测试用例级别"
    exit 0
}

# 解析长选项
parse_long_option() {
    case "$1" in
        --host)
            SERVER_HOST="$2"
            ;;
        --html)
            REPORT_HTML="$2"
            ;;
        --task-output-path)
            TASK_OUTPUT_PATH="$2"
            ;;
        --pd-separation-flag)
            PD_SEPARATION_FLAG="$2"
            ;;
        --model-path)
            MODEL_PATH="$2"
            ;;
        --vllm-env)
            VLLM_ENV_PATH="$2"
            ;;
        --mockmodel-env)
            MOCKMODEL_ENV_PATH="$2"
            ;;
        --test-level)
            TEST_LEVEL="$2"
            ;;
    esac
    return 0
}

# 解析选项
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            ;;
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

# 打印当前配置
echo "==== 当前配置 ===="
echo "SERVER_HOST: $SERVER_HOST"
echo "REPORT_HTML: $REPORT_HTML"
echo "TASK_OUTPUT_PATH: $TASK_OUTPUT_PATH"
echo "PD_SEPARATION_FLAG: $PD_SEPARATION_FLAG"
if [ "$PD_SEPARATION_FLAG" = "0" ]; then
    echo "MODEL_PATH: $MODEL_PATH"
    echo "VLLM_ENV_PATH: $VLLM_ENV_PATH"
    echo "MOCKMODEL_ENV_PATH: $MOCKMODEL_ENV_PATH"
fi
echo "TEST_LEVEL: $TEST_LEVEL"
echo "=================="

log_info "部署vllm在线服务"
touch ${TASK_OUTPUT_PATH}/server.log
# 现状：
# PD不分离模式下，需要调用start_tests.sh启动服务后再执行用例。
# PD分离模式下，需要先启动服务再调用start_tests.sh执行用例。
if [ "${PD_SEPARATION_FLAG}" = "0" ]; then
    source ${VLLM_ENV_PATH}
    source ${MOCKMODEL_ENV_PATH}
    nohup bash start_vllm_server/deepseek_dp8_tp1_ep1.sh ${MODEL_PATH}/ &> ${TASK_OUTPUT_PATH}/server.log &

    log_info "进入循环等待服务启动标识符出现"
    cost=0
    interval=10
    endtime=300
    while true; do
        if [ $cost -gt $endtime ]; then
            log_error "等待服务启动时间超过${endtime}秒，退出循环"
            log_error "服务启动日志结尾部分内容:"
            tail -n 50 ${TASK_OUTPUT_PATH}/server.log
            exit 1
        fi
        if grep -q "Application startup complete" ${TASK_OUTPUT_PATH}/server.log; then
            log_info "服务已启动，继续执行用例"
            log_info "服务启动日志开头部分内容:"
            head -n 50 ${TASK_OUTPUT_PATH}/server.log
            break
        elif grep -q "NPU out of memory" ${TASK_OUTPUT_PATH}/server.log; then
            log_error "服务启动失败，NPU out of memory"
            log_error "服务启动日志结尾部分内容:"
            tail -n 50 ${TASK_OUTPUT_PATH}/server.log
            exit 1
        else
            log_warning "服务启动中，等待${interval}秒"
            sleep ${interval}
            cost=$((cost + interval))
        fi
    done
else
    # TODO: 待一键拉起PD分离服务就绪后，在这里拉起服务。
    echo "默认PD分离服务已启动"
fi

pytest -vsra --disable-warnings -m "${TEST_LEVEL}" \
--html=${REPORT_HTML} \
--ignore=st/cloud_tests_imp \
--ignore=st/test_edit_distance/test_edit_distance.py \
./st \
--host="${SERVER_HOST}" \
--output_dir="${TASK_OUTPUT_PATH}" \
--tokenizer="${MODEL_PATH}"

test_result=$?
# 检查退出状态码，如果不为0则表示有测试失败
if [ $test_result -ne 0 ]; then
    log_error "测试失败"
    exit 1
else
    echo "测试例100%通过"
fi

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_PATH/
log_info "当前目录: $SCRIPT_PATH"
test_qwen_result=$(bash test_qwen.sh --task-output-path ${TASK_OUTPUT_PATH})
if [ $test_qwen_result -ne 0 ]; then
    log_error "qwen脚本执行失败，返回值为: $test_qwen_result"
    exit 1
else
    echo "qwen脚本执行成功"
fi
# log_info "停止vllm服务"
# pkill -f python3 || true