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
def test_start_api_service(host):
    if check_service_status(host):
        logging.info("Service started successfully.")
    else:
        assert False, "Service started failed."

    url = f"http://{host}/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "deepseek",
        "prompt": ["The future of AI is", "Tell me a joke", "Who is the president of US?"],
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 1,
        "top_k": -1
    }

    response = requests.post(url, headers=headers, json=data)

    assert response.status_code == 200


if __name__ == "__main__":
    pytest.main()

