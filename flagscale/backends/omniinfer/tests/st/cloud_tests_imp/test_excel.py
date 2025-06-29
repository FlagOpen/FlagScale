# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# -*- coding: UTF-8 -*-
import time
import traceback

import pytest
from tools.http_tool import send_request_json, check_result
from tools import config
from tools.logger_tool import logger


class TestModel:

    @pytest.mark.level0
    @pytest.mark.parametrize("data", config.cases_l0, ids=config.ids_l0)
    def test_level0(self, data, args):
        logger.info(data)
        repeat = int(data["重复"]) if data["重复"] else 1
        logger.info(f"重复执行：{data['重复']}")

        for _ in range(repeat):
            start_time = time.time()
            try:
                response = send_request_json(data)
                check_result(data, response)
                end_time = time.time()
                logger.info("用例[%s]执行耗时: %s", data["用例名称"], end_time - start_time)
            except Exception as e:
                logger.info("用例[%s]执行耗时: %s", data["用例名称"], time.time() - start_time)
                logger.error("用例[%s]执行异常: %s, %s", data["用例名称"], str(e), traceback.format_exc())
                assert False, str(e)

    @pytest.mark.level1
    @pytest.mark.parametrize("data", config.cases_l1, ids=config.ids_l1)
    def test_level1(self, data, args):
        repeat = int(data["重复"]) if data["重复"] else 1
        logger.info(data)
        logger.info(f"重复执行：{data['重复']}")

        for _ in range(repeat):
            start_time = time.time()
            try:
                response = send_request_json(data)
                check_result(data, response)
                logger.info("用例[%s]执行耗时: %s", data["用例名称"], time.time() - start_time)
            except Exception as e:
                logger.info("用例[%s]执行耗时: %s", data["用例名称"], time.time() - start_time)
                logger.error("用例[%s]执行异常: %s, %s", data["用例名称"], str(e), traceback.format_exc())
                assert False, str(e)


if __name__ == '__main__':
    pytest.main(
        ["-s", "-n", "1", "test_excel.py", "--html=report.html",
         "--arg=DeepSeek-v3,http://7.242.107.61:9000/v1/chat/completions"])
