# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import os
from copy import deepcopy

from omni.accelerators.pd.ranktable.device import Server, Device

LOCAL_RANK_TABLE_ENV = "RANK_TABLE_FILE_PATH"


class LocalInfo:

    def __init__(self):
        self.decode_server_ip_list = os.getenv("LOCAL_DECODE_SERVER_IP_LIST", None).split(',')
        self._rank_table_info = self.get_ranktable_dict()
        self.server = self.init_server()

    @staticmethod
    def get_ranktable_dict():
        env_path = os.getenv(LOCAL_RANK_TABLE_ENV, None)
        if env_path is None:
            raise ValueError(
                f"Env `{LOCAL_RANK_TABLE_ENV}` is required for pd separate.")

        with open(env_path, 'r', encoding='utf-8') as f:
            rank_table = json.load(f)

        return rank_table

    def init_server(self):
        server_list = []
        results_device_list = []
        for server_info in self._rank_table_info["server_list"]:
            server_list.append(Server(server_info))
            for device_info in server_info.get("device"):
                results_device_list.append(Device(device_info))

        server_list[0].device_list = results_device_list

        if os.environ.get("ROLE") == "prefill":
            return server_list[0]

        results_device_list = []

        rank_id = 0
        for ip in self.decode_server_ip_list:
            success = False
            for server_info in self._rank_table_info["server_list"]:
                server_ip = server_info.get("server_ip")
                if ip == server_ip:
                    for device_info in server_info.get("device"):
                        device_info_tmp = deepcopy(device_info)
                        device_info_tmp["rank_id"] = str(rank_id)
                        rank_id += 1
                        results_device_list.append(Device(device_info_tmp))
                    success = True

            if not success:
                raise ValueError(
                    f"({ip}) NOT in local_ranktable, please check the DECODE_SERVER_IP_LIST env)")

        server_list[0].device_list = results_device_list
        return server_list[0]