# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# -*- coding: UTF-8 -*-
import pytest
import os
import datetime

from tools.logger_tool import logger


def add_timestamp_to_filename(filename):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 获取当前时间并格式化为字符串
    new_filename = f"{filename}_{timestamp}"
    return new_filename


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--url', type=str, required=True, help="服务请求全路径")
    parser.add_argument('--model_name', type=str, required=True, help="模型名称")
    parser.add_argument('--max_fail', '-m', type=int, default='500',
                        help='多少用例执行失败任务停止，服务端异常的话任务能快速停止')
    parser.add_argument('--parallel_num', '-n', type=int, default='25',
                        help='并发执行用例')
    parser.add_argument('--case_type', '-t', type=str, choices=['deepseek_r1', 'deepseek_v3'],
                        help='用例类型')
    parser.add_argument('--case_level', type=str, default='level0',
                        help='用例级别')
    parser.add_argument('--case_time', type=str, choices=['fast', 'slow'], default='fast',
                        help='用例类型')
    parser.add_argument('--model_architecture', type=str, choices=['tp16', 'tp16_mtp', 'pd'], default='pd',
                        help='部署形态')
    parser.add_argument('--api_key', type=str, help='秘钥')

    args = parser.parse_args()
    logger.info(f"启动参数：\n{args}")
    os.environ["case_type"] = args.case_type
    os.environ["model_architecture"] = args.model_architecture
    if args.case_time == 'fast':
        os.environ["case_time"] = "快任务"
    else:
        os.environ["case_time"] = "慢任务"

    report_name = add_timestamp_to_filename(f"report_{args.case_type}_{args.case_time}")

    # 获取当前时间并格式化为字符串
    current_time = time.strftime("%Y%m%d%H%M%S")
    pytest.main(
        ["-vs", f"-m={args.case_level}", "-n", f"{args.parallel_num}", "./", '--capture=sys',
         "--maxfail",
         f"{args.max_fail}",
         f"--html=./report/{report_name}.html",
         f"--args={args.url},{args.model_name},{args.api_key}", "--dist=load",
         f"--log-file=./report/pytest_{current_time}.log"])

    logger.warning("华为云功能用例执行结束")
