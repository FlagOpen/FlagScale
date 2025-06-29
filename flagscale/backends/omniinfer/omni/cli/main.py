#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import argparse
import subprocess
import yaml
import requests
import json
import os
from omni.cli.config_transform import transform_deployment_config

def load_config(config_path):
    """加载并解析 YAML 配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_ansible_playbook_with_config(config_path):
    """使用配置文件作为 inventory 执行 Ansible Playbook"""
    transform_deployment_config(config_path)
    command = f"ansible-playbook -i omni_infer_inventory.yml omni_infer_server.yml"
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"部署失败: stdout:{stdout.decode()} stderr:{stderr.decode()}")
    else:
        print(f"部署成功: {stdout.decode()}")


def check_service_health(config):
    """执行服务健康检查"""
    try:
        # 解析代理配置
        proxy_host = config['deployment']['proxy']['host']
        proxy_port = config['deployment']['proxy']['listen_port']
        model_path = config['services']['model_path']

        # 构造请求 URL
        url = f"http://{proxy_host}:{proxy_port}/v1/completions"

        # 构造请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY"  # 替换为实际 API 密钥
        }

        # 构造请求体
        payload = {
            "model": model_path,
            "prompt": "Alice is ",
            "max_tokens": 50,
            "temperature": 0
        }

        # 发送请求
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        # 处理响应
        if response.status_code == 200:
            print("✅ 服务健康检查通过")
            print(f"响应内容: {response.json()}")
            return True
        else:
            print(f"❌ 服务异常 (状态码: {response.status_code})")
            print(f"错误信息: {response.text}")
            return False

    except KeyError as e:
        print(f"❌ 配置错误: 缺少必要的配置项 {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 网络错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Omni Inference 服务管理")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 新增 serve 子命令
    serve_parser = subparsers.add_parser("serve", help="部署推理服务")
    serve_parser.add_argument("config", help="配置文件路径")

    # 新增 status 子命令
    status_parser = subparsers.add_parser("status", help="服务健康检查")
    status_parser.add_argument("--config", default="omni_infer_deployment.yml", help="配置文件路径")

    args = parser.parse_args()

    if args.command == "serve":
        # 执行 Ansible 部署
        print(f"🚀 开始部署服务，使用配置文件: {args.config}")
        run_ansible_playbook_with_config(args.config)
    elif args.command == "status":
        # 执行健康检查
        print(f"🔍 开始服务健康检查，使用配置文件: {args.config}")
        config = load_config(args.config)
        check_service_health(config)


if __name__ == "__main__":
    main()
