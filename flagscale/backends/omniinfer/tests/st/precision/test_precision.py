# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# All rights reserved.

import logging
import os
import subprocess

import requests
import pytest
from tests.mark_utils import arg_mark
from tests.st.scripts.utils import check_service_status


@arg_mark(['platform_ascend910b'], 'level0')
def test_precision(host):
    if check_service_status(host):
        logging.info("Service started successfully.")
    else:
        assert False, "Service started failed."

    url = f"http://{host}/v1/completions"
    headers = {"Content-Type": "application/json"}
    json_data = {
        "model": "deepseek",
        "prompt": ["计算365乘以24"],
        "max_tokens": 50,
        "temperature": 0,
        "top_p": 1,
        "top_k": -1
    }

    response = requests.post(url, headers=headers, json=json_data)

    if response.status_code == 200:
        data = response.json()
        # 检查是否有内容返回
        assert data, "API 返回内容为空"
        if not data:
            assert False, "API 返回内容为空"
        else:
            logging.info(f"API 返回内容不为空,data:{data}")

        # deepseek_v3检查返回内容中是否有 'think' 标签
        # if 'think' in data:
        #     logging.info("返回内容中包含 'think' 标签")
        # else:
        #     assert False, "返回内容中不包含 'think' 标签"
        #
        # # 检查 'reasoning_content' 内容是否为空
        # if 'reasoning_content' in data:
        #     reasoning_content = data['reasoning_content']
        #     if reasoning_content:
        #         logging.info("reasoning_content 不为空")
        #     else:
        #         assert False, "reasoning_content 为空"
        # else:
        #     assert False, "'reasoning_content' 标签不存在"
    else:
        assert False, f"请求失败，状态码: {response.status_code}， 响应结果: {response.content}"


TEST_PRECISION = os.getenv("TEST_PRECISION", False)


@arg_mark(['platform_ascend910b'], 'level1')
def test_precision_level1(host):
    if not TEST_PRECISION:
        # 由于每日构建部署的模型不足以运行精度用例，暂时关闭图灵进度用例
        return
    print("执行精度用例")
    curdir = os.getcwd()
    curfiledir = os.path.dirname(__file__)
    imp_path = os.path.join(curfiledir, '../simple_evals')
    os.chdir(imp_path)
    p = subprocess.Popen(["/bin/bash", "-c", f'./run_simple_eval.sh "{host}" level1'],
                         stdout=subprocess.PIPE,  # 捕获标准输出
                         stderr=subprocess.PIPE,  # 可选：捕获错误输出
                         text=True,  # 直接返回字符串（Python 3.7+）
                         cwd=imp_path,
                         encoding="utf-8")
    timeout = None
    out, err = p.communicate(timeout=timeout)
    p.wait()
    os.chdir(curdir)
    print(f"out is :{out}")
    print(f"err is :{err}")


if __name__ == "__main__":
    pytest.main()
