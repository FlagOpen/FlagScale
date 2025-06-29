# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# All rights reserved.

import logging

import requests
import pytest
from tests.mark_utils import arg_mark
from tests.st.scripts.utils import check_service_status


@arg_mark(['platform_ascend910b'], 'level0')
def test_precision_chat_level0(host, served_model_name):
    if check_service_status(host):
        logging.info("Service started successfully.")
    else:
        assert False, "Service started failed."

    url = f"http://{host}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    json_data = {
        "messages": [
            {
                "role": "user",
                "content": "赵女士买了一些水果和小食品准备去看望一个朋友，谁知，这些水果和小食品被他的儿子们偷吃了，但她不知道是哪个儿子。，为此，赵女士非常生气，就盘问4个儿子谁偷吃了水果和小食品。老大说道：“是老二吃的。”老二说道：“是老四偷吃的。”老三说道：“反正我没有偷吃。”老四说道：“老二在说谎。”这4个儿子中只有一个人说了实话，其他的3个都在撒谎。那么，到底是谁偷吃了这些水果和小食品？"
            }
        ],
        "model": served_model_name,
        "temperature": 0,
        "max_tokens": 1500
    }

    response = requests.post(url, headers=headers, json=json_data)

    if response.status_code == 200:
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as err:
            assert False, f"回答格式不正确：{str(err)}"
        # 取出</think>标签后的内容作为模型输出
        parts = content.split('</think>')
        value = parts[1].strip() if len(parts) > 1 else parts[0].strip()

        if "老三" in value and "偷吃" in value:
            assert True
        else:
            assert False, "回答不符合预期"
    else:
        assert False, f"请求失败，状态码: {response.status_code}， 响应结果: {response.content}"


if __name__ == "__main__":
    pytest.main()
