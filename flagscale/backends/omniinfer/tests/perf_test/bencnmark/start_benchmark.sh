#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

set -e

# 获取当前脚本的绝对路径并进入
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_PATH

# 导入日志工具函数
source "$SCRIPT_PATH/../../log_utils.sh"
OUTPUT_DIR=${OUTPUT_DIR-"output"}
TASK_OUTPUT_PATH=${OUTPUT_PATH}/${OUTPUT_DIR}
LEVEL_INPUT=level0
FORWARD_TIME_SEND=0.01
MODEL_NAME=deepseek

export PYTHONPATH=${PYTHONPATH}:$SCRIPT_PATH/..:$SCRIPT_PATH/../../../tools/llm_evaluation/benchmark_tools

if [ ! -d "$TASK_OUTPUT_PATH" ]; then
    mkdir -p ${TASK_OUTPUT_PATH}
fi
ENV_PATH=$SCRIPT_PATH/../../env.sh
MOCK_MODEL_ENV_PATH=$SCRIPT_PATH/mock_model_env.sh

function start_benchmark {
  log_info "vllm服务化启动"
  touch ${TASK_OUTPUT_PATH}/server.log
  MODEL_PATH="${INPUT_PATH}/model/DeepSeek-V3-w8a8-0423"
  log_info "指定模型路径: ${MODEL_PATH}"
  touch ${TASK_OUTPUT_PATH}/server.log
  source ${ENV_PATH}
  log_info "FORWARD_TIME 的值: ${FORWARD_TIME}"
  nohup bash $SCRIPT_PATH/../../start_vllm_server/qwen_dp8_tp1.sh ${MODEL_PATH}/ &> ${TASK_OUTPUT_PATH}/server.log &

  log_info "进入循环等待服务启动标识符出现"
  cost=0
  interval=10
  endtime=300
  while true; do
      if [ $cost -gt $endtime ]; then
          log_error "等待服务启动时间超过${endtime}秒，退出循环"
          log_error "服务启动日志结尾部分内容:"
          tail -n 50 ${TASK_OUTPUT_PATH}/server.log
          break
      fi
      if grep -q "Application startup complete" ${TASK_OUTPUT_PATH}/server.log; then
          log_info "服务已启动，继续执行用例"
          log_info "服务启动日志开头部分内容:"
          head -n 50 ${TASK_OUTPUT_PATH}/server.log
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

  pytest -vsra --disable-warnings -m "platform_ascend910b and ${LEVEL_INPUT}" \
  ./perf_test/bencnmark/test_mock_model_benchmark.py \
  --output_dir=${TASK_OUTPUT_PATH} \
  --except_time=0.003 \
  --forward_time=${FORWARD_TIME_SEND} \
  --tokenizer=${MODEL_PATH} \
  --served_model_name=${MODEL_NAME}

  log_info "停止vllm服务"
  pkill -f python3
}

# 定义清理函数
cleanup() {
    log_info "停止vllm服务"
    pkill -f python3
    exit $?
}

# 检查参数是否为空
if [ -z "$1" ]; then
  log_info "请提供一个参数，例如 level0 或 level1"
  exit 1
fi

# 捕获退出信号并执行清理函数
trap cleanup EXIT

# 根据参数执行不同的步骤
case "$1" in
  level0)
    log_info "开始执行level0的FORWARD_TIME=10ms性能测试用例"
    LEVEL_INPUT=level0
    start_benchmark
    ;;
  level1)
    log_info "开始执行level1的FORWARD_TIME=10ms性能测试用例"
    LEVEL_INPUT=level1
    start_benchmark
    log_info "开始执行FORWARD_TIME=45ms性能测试用例"
    ENV_PATH=${MOCK_MODEL_ENV_PATH}
    FORWARD_TIME_SEND=0.045
    start_benchmark
    ;;
  *)
    log_info "无效的参数，请使用 level0 或 level1"
    exit 1
    ;;
esac


exit 0

