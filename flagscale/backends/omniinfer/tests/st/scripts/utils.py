# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# All rights reserved.

import json
import requests


def check_service_status(url="http://127.0.0.1:8000/v1/models"):
    # 如果 URL 已经包含协议部分，直接使用
    if url.startswith("http://") or url.startswith("https://"):
        processed_url = url
    else:
        # 如果 URL 不包含协议部分，添加默认的 "http://" 和 "/v1/models"
        processed_url = f"http://{url}/v1/models"

    try:
        response = requests.get(processed_url)
        return response.status_code == 200
    except requests.RequestException:
        return False
