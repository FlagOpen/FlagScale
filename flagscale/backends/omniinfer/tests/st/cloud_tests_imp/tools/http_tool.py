# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# -*- coding: UTF-8 -*-

import requests
import json
import os
import re

from tools.logger_tool import logger


# 发送一些非json的数据的这个方法可以
def send_request(data, headers):
    try:
        url = os.getenv("url")
        case_name = data["用例名称"]
        tmp = data["配置项"].replace("$MODEL", os.getenv("model_name"))
        logger.info(f"用例{case_name},body: {tmp}")

        # payload = tmp.encode("utf-8").decode("latin1")
        payload = tmp.encode("utf-8")
        response = requests.request("POST", url, data=payload, headers=headers, verify=False, timeout=80)
        logger.info(f"用例{case_name},返回码: {response.status_code}")
        logger.info(f"用例{case_name},返回体: {response.text}")
    except Exception as e:
        logger.exception(e)
    return response


def send_request_json(data):
    case_name = data["用例名称"]
    url = os.getenv("url")
    api_key = os.getenv("api_key")
    headers = {'Content-Type': 'application/json'}
    if api_key != "null":
        headers['Authorization'] = f'Bearer {api_key}'
    try:
        payload = json.loads(data["配置项"])
        payload["model"] = os.getenv("model_name")
    except Exception as e:
        logger.exception(e)
        # body非json的时候走这个方法
        return send_request(data, headers)

    logger.info(f"用例{case_name},body: {payload}")
    payload = json.dumps(payload)

    response = requests.request("POST", url, data=payload, headers=headers, verify=False, timeout=80)
    logger.info(f"用例{case_name},返回码: {response.status_code}")
    logger.info(f"用例{case_name},返回体: {response.text}")

    return response


def send_request_data(data, case_name):
    url = os.getenv("url")
    api_key = os.getenv("api_key")
    headers = {'Content-Type': 'application/json'}
    data["model"] = os.getenv("model_name")
    if api_key != "null":
        headers['Authorization'] = f'Bearer {api_key}'

    payload = json.dumps(data)
    logger.info(f"用例{case_name},body体: {data}")
    response = requests.request("POST", url, data=payload, headers=headers, verify=False)

    logger.info(f"用例{case_name},返回码: {response.status_code}")
    logger.info(f"用例{case_name},返回体: {response.text}")

    return response


def send_request_stream(data, case_name):
    response = send_request_data(data, case_name)
    assert response.status_code == 200
    pattern = r'data:\s*(.*)'
    results = re.findall(pattern, response.text)
    # 删除DONE
    results.pop(-1)
    return results


def check_result(data, response):
    assert response.status_code == data["返回码"], response.status_code

    if data["规则"]:
        check_rules = data["规则"].split("&")
        for check_rule in check_rules:
            if check_rule == "包含":
                datas = str(data["包含"]).split("我是分隔符")
                for tmp in datas:
                    tmp = tmp.replace("$MODEL", os.getenv("model_name"))
                    if not tmp in response.text:
                        assert False, f"{tmp} not in response"
                logger.info(f"用例{data['用例名称']}包含,{datas}")
            elif check_rule == "不包含":
                datas = str(data["不包含"]).split("我是分隔符")

                for tmp in datas:
                    if tmp in response.text:
                        assert False, f"{tmp}  in response"
                logger.info(f"用例{data['用例名称']}不包含,{datas}")
            elif check_rule == "正则匹配":
                _check_field_value(data["正则匹配"], response)
            elif check_rule == "字段匹配":
                response = response.json()
                rules = data['字段匹配'].slpit("我是分割符")
                for rule in rules:
                    check_res = eval(rule)
                    assert check_res, rule
            else:
                pass


# 字段匹配以表格里的换行为标准
def _check_field_value(pattern, response):
    datas = pattern.split("我是分隔符")
    for data in datas:
        # 正则表达式匹配,匹配换行
        match = re.search(data, response.text, flags=re.DOTALL)
        if match:
            logger.info(f'{data},匹配的结果:' + match.group(0))
            continue

        else:
            assert False, "字段内容匹配错误"


CHUNK_SIZE = 1024
CHUNK_LINE_START_PREFIX = "data: "
CHUNK_LINE_END_TAG = "[DONE]"
from requests import session


def send_request_chunk(data, case_name):
    url = os.getenv("url")
    api_key = os.getenv("api_key")
    headers = {'Content-Type': 'application/json'}
    data["model"] = os.getenv("model_name")
    if api_key != "null":
        headers['Authorization'] = f'Bearer {api_key}'
    logger.info(f"用例{case_name},body体: {data}")
    try:
        response = requests.Session().post(
            url,
            headers=headers,
            json=data,
            stream=True,
            verify=False,
            timeout=None
        )
    except Exception as e:
        logger.exception("这里报错")
        logger.exception(e)
        return None

    results = []
    response.raise_for_status()
    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
        if chunk:
            chunk_str = chunk.decode('utf-8')
            lines = chunk_str.split('\n')
            for line in lines:
                if line.startswith(CHUNK_LINE_START_PREFIX):
                    logger.info(line)
                    # 最后的一个chunk是 data: [Done], 没有choices
                    if line.endswith(CHUNK_LINE_END_TAG):
                        continue
                    # 解析 JSON 数据
                    json_data = json.loads(line[len(CHUNK_LINE_START_PREFIX):])
                    results.append(json_data)
    return results
