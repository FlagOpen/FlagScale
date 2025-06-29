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

import yaml
import chardet
import re
import logging
import os
import sys

# Configure the log system 
def setup_logging(log_file='omni_cli.log', log_level=logging.INFO):
    """
    Configure the log system to overwrite the previous log each time it is executed. 
    
    Parameters:
        log_file (str): Log file path, default is 'omni_cli.log'
        log_level (int): Log level, default is logging.INFO
    """
    # Get root logger 
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove all existing processors (to avoid duplicate logs) 
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create a log formatter 
    formatter = logging.Formatter(
        '%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create file processor (overwrite mode) 
    try:
        # Ensure that the log directory exists. 
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Use overwrite mode ('w') instead of append mode ('a') 
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as e:
        sys.stderr.write(f"Unable to create log file: {e}\n")
        # Go back to console logs 
        pass
    
    # Create console processor (output to terminal) 
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add startup log 
    logger.info("=" * 80)
    logger.info(f"Log system initialization completed - Log level: {logging.getLevelName(log_level)}")
    logger.info(f"Log files: {os.path.abspath(log_file)}")
    logger.info("=" * 80)
    
    return logger

def detect_file_encoding(file_path):
    """Automatically detect file encoding"""
    with open(file_path, 'rb') as file:  # Read in binary mode
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def parse_host_overrides(logger, decode_group):
    """
    Analyze the host coverage configuration in the decode group. 
    Supported formats: 
        host 127.0.0.6:
            master_port: "8000"
            base_api_port: "8010"
            private_key: "/workspace/pem/keypair-04.pem"
    """
    overrides = {}
    host_overrides = {}
    key_map = {
        'user': 'ansible_user',
        'master_port': 'node_port',
        'base_api_port': 'api_port',
        'ascend_rt_visible_devices': 'ascend_rt_visible_devices',
        'private_key': 'ansible_ssh_private_key_file',
        'password': 'ansible_password'
    }
    
    for key, value in decode_group.items():
        host = key.split(" ")[0]
        if host != 'host':
            continue

        ip = key.split(" ")[1]

        for keyName, hostConfig in value.items():
            if new_key := key_map.get(keyName):
                host_overrides[new_key] = hostConfig
            else:
                logger.error("There are configurations that do not meet the requirements! The configurations that do not meet the requirements are: %s", keyName)
                return None

        overrides[ip] = host_overrides
        host_overrides = {}

    return overrides

def transform_config_for_inventory(logger, input_data):
    """The core function for executing configuration conversion"""
    # Extract basic information
    docker_image = input_data['deployment']['docker_image']
    required_keys = {'user', 'private_key', 'master_port', 'base_api_port', 'ascend_rt_visible_devices'}
    
    # Building infrastructure
    output = {
        'all': {
            'vars': {
                'docker_image': docker_image,
                'ansible_ssh_common_args': '-o StrictHostKeyChecking=no -o IdentitiesOnly=yes'
            },
            'children': {
                'P': {'hosts': {}},
                'D': {'hosts': {}},
                'C': {'hosts': {}}
            }
        }
    }
    
    # Handle the prefill section 
    prefill = input_data['deployment']['prefill']
    p_hosts = output['all']['children']['P']['hosts']

    # Group processing 
    index = 0
    for group_valus in prefill.values():
        host_key = f'p{index}'
        hosts_list = [h.strip() for h in group_valus['hosts'].split(',')]
        if len(hosts_list) > 1:
            logger.error("Multiple machines cannot deploy one prefill instance at present!")
            return None

        missing_keys = required_keys - set(group_valus.keys())
        if missing_keys:
            logger.error("The group of prefill is missing the necessary configuration. The missing configuration is: %s", missing_keys)
            return None

        p_hosts[host_key] = {
            'ansible_host': hosts_list[0],
            'ansible_user': group_valus['user'],
            'ansible_ssh_private_key_file': group_valus['private_key'],
            'node_rank': f'{index}',
            'node_port': group_valus['master_port'],
            'api_port': group_valus['base_api_port'],
            'ascend_rt_visible_devices': group_valus['ascend_rt_visible_devices']
        }

        index+=1
    
    # Handle the decode section 
    if len(input_data['deployment']['decode']) > 1:
        logger.error("Multiple decode instances are not supported for now!")
        return None

    decode_group = input_data['deployment']['decode']['group1']
    missing_keys = required_keys - set(decode_group.keys())
    if missing_keys:
        logger.error("The group of decode is missing necessary configuration. The missing configuration is: %s", missing_keys)
        return None

    d_hosts = output['all']['children']['D']['hosts']
    host_overrides = parse_host_overrides(logger, decode_group)
    if host_overrides is None:
        return None

    hosts_list = [h.strip() for h in decode_group['hosts'].split(',')]
    
    # Create a configuration for each decode host 
    for i, host in enumerate(hosts_list):
        host_key = f'd{i}'
        
        # Use group configuration by default 
        host_config = {
            'ansible_host': host,
            'ansible_user': decode_group['user'],
            'ansible_ssh_private_key_file': decode_group['private_key'],
            'node_port': decode_group['master_port'],
            'api_port': decode_group['base_api_port'],
            'ascend_rt_visible_devices': decode_group['ascend_rt_visible_devices'],
            'role': 'S'  # Default slave server 
        }
        
        # First host special processing
        if i == 0:
            host_config['role'] = 'M'  # Master server

        if host in host_overrides.keys():
            host_config.update({
                k: v for k, v in host_overrides[host].items()
            })
        
        d_hosts[host_key] = host_config

    # Handle proxy part (convert to C group) 
    proxy = input_data['deployment']['proxy']
    c_hosts = output['all']['children']['C']['hosts']
    
    c_hosts['c0'] = {
        'ansible_host': proxy['host'],
        'ansible_user': proxy['user'],
        'ansible_ssh_private_key_file': proxy['private_key'],
        'node_port': proxy['listen_port']
    }
    
    return output

def transform_config_for_playbook(logger, input_data):
    """The core function for executing configuration conversion"""
    updates = {}
    if "docker_image" not in input_data['deployment']:
        logger.error("Missing global configuration: %s", "docker_image")
        return None
    if "model_path" not in input_data['services']:
        logger.error("Missing global configuration: %s", "model_path")
        return None
    if 'max_model_len' not in input_data['services']['prefill']:
        logger.error("Missing necessary configuration for prefill: %s", 'max_model_len')
        return None
    if 'max_model_len' not in input_data['services']['decode']:
        logger.error("Missing necessary configuration for decoding: %s", 'max_model_len')
        return None

    updates["DOCKER_IMAGE_ID"] = input_data['deployment']['docker_image']
    updates["MODEL_PATH"] = input_data['services']['model_path']
    updates["MODEL_LEN_MAX_PREFILL"] = input_data['services']['prefill']['max_model_len']
    updates["MODEL_LEN_MAX_DECODE"] = input_data['services']['decode']['max_model_len']

    return updates

def update_yml_file(logger, updates, file_path):  
    # Automatic detection of file encoding 
    encoding = detect_file_encoding(file_path)
    
    # Read the file using the detected encoding 
    if not encoding:
        logger.warning("The encoding of the omni_infer_server.yml file is unknown. UTF-8 is used by default.")
        encoding = 'utf-8'
    else:
        logger.info(f"Detected omni_infer_server.yml file encoding: {encoding}")
    
    with open(file_path, 'r', encoding=encoding, errors='replace') as file:
        content = file.read()
    
    # Create a regular expression pattern for each key 
    patterns = {
        key: re.compile(rf'^(\s*{key}:\s*)"([^"]+)"', re.MULTILINE)
        for key in updates
    }
    
    # Replace the value of each target key. 
    for key, new_value in updates.items():
        content = patterns[key].sub(rf'\1"{new_value}"', content)
    
    # Write the file back using the same encoding as the original file 
    with open(f'{os.getcwd()}/omni_infer_server.yml', 'w', encoding=encoding) as file:
        file.write(content)

def transform_deployment_config(config_path):
    """Main function, handles file input and output"""
    try:
        # Initialize the log system
        logger = setup_logging()
 
        # Automatically detect file encoding
        encoding = detect_file_encoding(config_path)

        # Read the file using the detected encoding 
        if not encoding:
            logger.warning("The encoding of the config_path file is unknown. UTF-8 is used by default.")
            encoding = 'utf-8'
        else:
            logger.info(f"Detected config_path file encoding: {encoding}")

        with open(config_path, 'r', encoding=encoding, errors='replace') as f:
            input_data = yaml.safe_load(f)
        
        # Execute inventory conversion 
        output_data = transform_config_for_inventory(logger, input_data)
        if output_data is None:
            return

        # Execute playbook transformation 
        playbookArges = transform_config_for_playbook(logger, input_data)
        if playbookArges is None:
            return

        # Get the absolute path of the current script 
        current_file = os.path.abspath(__file__)

        # Get the directory where the current script is located 
        omni_infer_server_path = os.path.dirname(current_file)
        update_yml_file(logger, playbookArges, f"{omni_infer_server_path}/omni_infer_server.yml")
        
        # Write to output file
        with open('omni_infer_inventory.yml', 'w') as f:
            yaml.dump(output_data, f, sort_keys=False, default_flow_style=False, indent=2)
        
        logger.info("Configuration file conversion successful! ")
        logger.info(f"Convert the output files to {os.getcwd()}/omni_infer_inventory.yml and {os.getcwd()}/omni_infer_server.yml ")
    
    except FileNotFoundError:
        logger.critical("Cannot find config_path file! ")
    except KeyError as e:
        logger.critical(f"The input file is missing a required key  - {e}")
    except Exception as e:
        logger.critical(f"An unknown error occurred: {e}")
