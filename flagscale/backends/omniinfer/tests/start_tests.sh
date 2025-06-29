#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

set -e

TEST_LEVEL=${TEST_LEVEL-"level0"}
PD_SEPARATION_PORT=${PD_SEPARATION_PORT-"8001"}
PD_SEPARATION_FLAG=${PD_SEPARATION_FLAG-"0"} # 0 表示不分离，1 表示分离
# 获取当前脚本的绝对路径并进入
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_PATH
chmod -R +x ./*

# 导入日志工具函数
source "$SCRIPT_PATH/log_utils.sh"

OUTPUT_DIR=${OUTPUT_DIR-"output"}
INPUT_PATH=${INPUT_PATH-"/home/ma-user/modelarts/inputs/data_url_0"}
OUTPUT_PATH=${OUTPUT_PATH-"/home/ma-user/modelarts/outputs/train_url_0"}
TASK_OUTPUT_PATH=${OUTPUT_PATH}/${OUTPUT_DIR}

# 导入benchmark_tools的路径
export PYTHONPATH=${PYTHONPATH}:$SCRIPT_PATH/..:$SCRIPT_PATH/../tools/llm_evaluation/benchmark_tools

if [ -d "${TASK_OUTPUT_PATH}" ]; then
    log_info "门禁输出目录存在"
else
    log_info "新建门禁输出目录"
    mkdir -p ${TASK_OUTPUT_PATH}
fi

MODEL_PATH="${INPUT_PATH}/model/DeepSeek-V3-w8a8-0423"
log_info "指定模型路径: ${MODEL_PATH}"

echo "==== start_tests.sh 当前配置 ===="
echo "PD_SEPARATION_FLAG: $PD_SEPARATION_FLAG"
echo "TEST_LEVEL: $TEST_LEVEL"
echo "================================"

# 此用例本身与PD分离无关，需要另起离线单机服务
if [ "${TEST_LEVEL}" != "level0" ] && [ "${PD_SEPARATION_FLAG}" = "0" ]; then
    log_info "Test edit distance"
    cd $SCRIPT_PATH
    REPORT_NAME="pytest_edit_distance_report.html"
    unset RANDOM_MODE
    pytest -vsra --disable-warnings -m "${TEST_LEVEL}" \
    --html=${TASK_OUTPUT_PATH}/${REPORT_NAME} \
    ./st/test_edit_distance/test_edit_distance.py
    log_info "停止vllm服务"
    pkill -f python3 || true
fi

cd $SCRIPT_PATH
if [ "${PD_SEPARATION_FLAG}" = "0" ]; then # PD不分离
    SERVER_HOST="127.0.0.1:8000"
    REPORT_NAME="pytest_report.html"
else # PD分离
    SERVER_HOST="127.0.0.1:${PD_SEPARATION_PORT}"
    REPORT_NAME="pytest_report_2p1d.html"
fi

log_info "st入口脚本"
bash st/start_st.sh --host "${SERVER_HOST}" \
    --html "${TASK_OUTPUT_PATH}/${REPORT_NAME}" \
    --task-output-path "${TASK_OUTPUT_PATH}" \
    --pd-separation-flag "${PD_SEPARATION_FLAG}" \
    --model-path "${MODEL_PATH}" \
    --vllm-env "${SCRIPT_PATH}/env.sh" \
    --mockmodel-env "${SCRIPT_PATH}/perf_test/bencnmark/mock_model_env.sh" \
    --test-level "${TEST_LEVEL}"
# TODO：bash perf_test/bencnmark/start_benchmark.sh 定长/变长