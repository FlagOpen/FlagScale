# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import subprocess
import yaml
import re
import os
import pytest
from tests.mark_utils import arg_mark
from datetime import datetime, timezone, timedelta


@arg_mark(['platform_ascend910b'], 'level1')
def test_variable_length(output_dir, forward_time, except_time, tokenizer, served_model_name, host, parallel_num):
    baseCmd = ['python', 'run_benchmark_tencent.py']
    if parallel_num is not None:
        baseCmd.extend(['--parallel-num', str(parallel_num)])
    if host is not None:
        update_base_url(host)
    eastern_8 = timezone(timedelta(hours=8))
    timestamp = datetime.now(tz=eastern_8).strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'output_{timestamp}.txt')

    try:
        # 将 stdout 和 stderr 合并到同一个文件
        with open(output_file, 'w') as f_output:
            result = subprocess.run(baseCmd, stdout=f_output, stderr=subprocess.STDOUT, text=True)
            print(result)
        if result.returncode == 0:
            print("变长用例执行成功")
        else:
            print("变长用例执行失败")
        print(f"输出已保存到 {output_file}")
    except Exception as e:
        # 将异常写入文件
        with open(output_file, 'a') as f_output:
            f_output.write(f"执行过程中发生异常: {str(e)}\n")
        print(f"执行过程中发生异常: {e}")
        return


def update_base_url(new_url):
    """
    更新YAML配置文件中的base_url字段，替换其中的IP地址和端口号。

    参数:
        new_address: 127.0.0.1:8000
    """
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 配置文件路径
    config_file = os.path.join(script_dir, 'providers.yaml')

    try:
        # 检查配置文件是否存在
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件 {config_file} 不存在。")

        # 读取YAML文件
        with open(config_file, 'r') as file:
            data = yaml.safe_load(file)

        # 正则表达式匹配base_url中的IP和端口号部分
        url_pattern = re.compile(r'(http://|https://)(.*?)(/.*)')
        # 遍历providers列表，修改base_url
        for provider in data.get('providers', []):
            if 'base_url' in provider:
                match = url_pattern.match(provider['base_url'])
                if match:
                    current_url = match.group(2)
                    if current_url == new_url:
                        return

                    # 替换url
                    update_url = f"{match.group(1)}{new_url}{match.group(3)}"
                    provider['base_url'] = update_url

        # 将修改后的内容写回YAML文件
        with open(config_file, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

    except yaml.YAMLError as e:
        print(f"YAML文件解析错误: {e}")
    except IOError as e:
        print(f"文件操作错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    pytest.main()
