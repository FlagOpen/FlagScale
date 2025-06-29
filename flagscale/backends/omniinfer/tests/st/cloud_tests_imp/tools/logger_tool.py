# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# -*- coding: UTF-8 -*-
import logging
import os.path
import sys


def init_logger():
    # 创建日志记录器
    print("======================init logger")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # 设置日志级别为 DEBUG

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 设置控制台日志级别为 DEBUG

    # 创建文件处理器
    # pid = os.getpid()
    # file_handler = logging.FileHandler(f'{pid}.log', mode='w', encoding='utf-8')
    # 追加模式多进程执行往一个文件里写会不会有并发问题？ mode='a'追加些就都能打印了
    if not os.path.exists("./report"):
        os.mkdir("./report")
    file_handler = logging.FileHandler('./report/app.log', mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)  # 设置文件日志级别为 ERROR

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 将格式设置到处理器
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


logger = init_logger()
