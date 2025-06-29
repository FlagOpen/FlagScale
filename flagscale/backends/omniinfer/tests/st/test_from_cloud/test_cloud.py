# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# All rights reserved.

import logging
import os
import requests
import subprocess
import pytest
from tests.mark_utils import arg_mark
from tests.st.scripts.utils import check_service_status


def test_from_cloud(host, cmd):
    if check_service_status(host):
        logging.info("Service started successfully.")
    else:
        logging.error("Service not started.")
        assert False

    curdir = os.getcwd()
    curfiledir = os.path.dirname(__file__)
    imp_path = os.path.join(curfiledir, '../cloud_tests_imp')
    os.chdir(imp_path)
    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,  # 捕获标准输出
                         stderr=subprocess.PIPE,  # 可选：捕获错误输出
                         text=True,  # 直接返回字符串（Python 3.7+）
                         cwd=imp_path,
                         encoding="utf-8")
    out, err = p.communicate()
    return_code = p.wait()
    os.chdir(curdir)
    print(f"out is :{out}")
    print(f"err is :{err}")
    tgt_str = "[100%] PASSED "
    tgt_str2 = "[100%] SKIPPED "

    assert tgt_str in out or tgt_str in err or tgt_str2 in out or tgt_str2 in err


@arg_mark(['platform_ascend910b'], 'level0')
def test_from_cloud_l0(host):
    test_from_cloud(host, ["/bin/bash", "-c", f'./run_test_cloud.sh "{host}" level0'])


@arg_mark(['platform_ascend910b'], 'level1')
def test_from_cloud_l1(host):
    test_from_cloud(host, ["/bin/bash", "-c", f'./run_test_cloud.sh "{host}" level1'])


if __name__ == "__main__":
    pytest.main()
