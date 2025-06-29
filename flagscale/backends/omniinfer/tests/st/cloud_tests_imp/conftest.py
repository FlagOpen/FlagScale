# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# -*- coding: UTF-8 -*-
import pytest
import os

from tools.excel_tool import read_excel_new

from tools import config
import logging

logger = logging.getLogger(__name__)
logger.info("conftet start =========================")
config.cases = read_excel_new(os.path.join(os.getcwd(), 'tools', "1.xlsx"))
config.ids = [case["用例名称"] for case in config.cases]
logger.info(config.ids)

# 读取L0用例
config.cases_l0 = [case for case in config.cases if str(case.get("用例级别", "0")).strip() == "0"]
config.ids_l0 = [case["用例名称"] for case in config.cases_l0]

# 读取L1用例
config.cases_l1 = [case for case in config.cases if str(case.get("用例级别", "1")).strip() == "1"]
config.ids_l1 = [case["用例名称"] for case in config.cases_l1]


def pytest_addoption(parser):
    parser.addoption("--args", action="store")


@pytest.fixture(scope="session")
def args(request):
    logger.info("args=============================")
    args = request.config.getoption("--args").split(",")
    logger.info(args)
    os.environ["url"] = args[0]
    os.environ["model_name"] = args[1]
    os.environ["api_key"] = args[2]
    return args
