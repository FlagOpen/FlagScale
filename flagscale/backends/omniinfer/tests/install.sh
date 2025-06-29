#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

set -e

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=${OUTPUT_DIR-"output"}
MOCKMODEL_FLAG=${MOCKMODEL_FLAG-"0"} # 0 不打MOCKMODEL的patch、1 打上MOCKMODEL的patch
export INPUT_PATH=/home/ma-user/modelarts/inputs/data_url_0
export OUTPUT_PATH=/home/ma-user/modelarts/outputs/train_url_0

source "${SCRIPT_PATH}/log_utils.sh"

# CI -> "level0" | Daily version -> "level0 or level1"
TEST_LEVEL=${TEST_LEVEL-"level0"}

# PD_SEPARATION_FLAG: 0 PD不分离，1 PD分离
PD_SEPARATION_FLAG=${PD_SEPARATION_FLAG-"0"}

if [ ${TEST_LEVEL} = "level0" ]; then
    PD_SEPARATION_FLAG="1"
fi

log_info "TEST_LEVEL: ${TEST_LEVEL}"
log_info "PD_SEPARATION_FLAG: ${PD_SEPARATION_FLAG}"

cd $SCRIPT_PATH

log_info "安装CI依赖 & 卸载旧版本"
pip config set global.timeout 100
pip config set global.retries 10
pip install -U pip -q
pip install -r requirements-ci.txt -q

log_info "拉vllm和vllm_ascend代码"
cd $SCRIPT_PATH/..
if [ "${TEST_LEVEL}" = "level0" ]; then
    if [ "${MOCKMODEL_FLAG}" = "0" ]; then
        log_info "Without Mock model"
        bash build/build.sh
    else
        log_info "With Mock model"
        bash build/build.sh --ci "1"
    fi
else
    bash build/build.sh
fi

log_info "安装vllm"
pip install $SCRIPT_PATH/../build/dist/vllm*

log_info "安装高性能torch_npu包"
cd $SCRIPT_PATH/..
mkdir pta
cd pta
tar -xf ${INPUT_PATH}/whl/pytorch_v2.5.1_py310.tar.gz
pip install torch_npu-2.5.1.dev20250519-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
