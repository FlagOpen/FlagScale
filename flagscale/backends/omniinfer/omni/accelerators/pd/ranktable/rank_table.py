# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import enum
import json
import os
from copy import deepcopy
from datetime import datetime
import pytz

from omni.accelerators.pd.ranktable.device import ServerGroup, Server, Device

GLOBAL_RANK_TABLE_ENV = "GLOBAL_RANK_TABLE_FILE_PATH"


class GroupType(enum.Enum):
    SCHEDULER = "0"
    PREFILL = "1"
    DECODE = "2"


GROUP_INDEX_TO_TYPE = {
    "0": GroupType.SCHEDULER,
    "1": GroupType.PREFILL,
    "2": GroupType.DECODE
}

GROUP_TYPE_TO_ROLE = {
    GroupType.SCHEDULER: "scheduler",
    GroupType.PREFILL: "prefill",
    GroupType.DECODE: "decode"
}


class GlobalRankTable:
    def __init__(self):
        self.prefill_pod_num = int(os.getenv("PREFILL_POD_NUM", "1"))
        self.decode_pod_num = int(os.getenv("DECODE_POD_NUM", "1"))

        self.global_decode_server_ip_list = [group.split(',')
                for group in os.getenv("GLOBAL_DECODE_SERVER_IP_LIST").split(';')]
        self.local_decode_server_ip_list = os.getenv("GLOBAL_DECODE_SERVER_IP_LIST").split(',')

        if len(self.global_decode_server_ip_list) != self.decode_pod_num:
            glo_d_ser_ip_num = len(self.global_decode_server_ip_list)
            d_pod_num = self.decode_pod_num
            raise ValueError(
                f"--global_dec_server_list size ({glo_d_ser_ip_num}) NOT equal DECODE_POD_NUM ({d_pod_num})")

        self._rank_table_info = self.get_ranktable_dict()
        self._device2info_dict = self.get_device2info_dict()
        self.group_dict = self.init_server_groups()
        self.group_dict_org = self.init_server_groups_org()

    @staticmethod
    def get_ranktable_dict():
        env_path = os.getenv(GLOBAL_RANK_TABLE_ENV, None)
        if env_path is None:
            raise ValueError(
                f"Env `{GLOBAL_RANK_TABLE_ENV}` is required for pd separate.")
        with open(env_path, 'r', encoding='utf-8') as f:
            rank_table = json.load(f)

        return rank_table

    @staticmethod
    def get_ranktable_dir():
        env_path = os.getenv(GLOBAL_RANK_TABLE_ENV, None)
        if env_path is None:
            raise ValueError(
                f"Env `{GLOBAL_RANK_TABLE_ENV}` is required for pd separate.")
        return os.path.dirname(env_path)

    def get_device2info_dict(self):
        device2info = {}
        server_group_list = self._rank_table_info.get("server_group_list")
        for server_group in server_group_list:
            server_list = server_group.get("server_list")
            for server in server_list:
                device_list = server.get("device", [])
                for device in device_list:
                    device_ip = device.get("device_ip")
                    device2info[device_ip] = device
        return device2info

    def get_server_list(self, node_list):
        server_list = []
        for node in node_list:
            server = {}
            device_list = []
            for device in node.device_ip_list:
                device_ip = device.device_ip
                device_info = self._device2info_dict.get(device_ip)
                device_list.append(device_info)
            server["server_id"] = node.host
            server["server_ip"] = node.host
            server["device"] = device_list
            server_list.append(server)
        return server_list

    def write_rank_table(self, prefill_node, decode_node):
        new_rank_table = deepcopy(self._rank_table_info)

        prefill_server_list = self.get_server_list(prefill_node)
        decode_server_list = self.get_server_list(decode_node)

        for server_group in new_rank_table.get('server_group_list'):
            group_id = server_group.get("group_id")
            if self.get_server_role() == "prefill":
                server_group["server_count"] = str(len(prefill_server_list))
                server_group["server_list"] = prefill_server_list
            elif self.get_server_role() == "decode":
                server_group["server_count"] = str(len(decode_server_list))
                server_group["server_list"] = decode_server_list

        content = json.dumps(new_rank_table, indent=4)
        local_time = datetime.now(pytz.UTC).strftime('%Y-%m-%d-%H%M%S')
        record_switch_file = os.path.join(
            self.get_ranktable_dir(), f'{local_time}.json')
        with open(record_switch_file, 'w+') as f:
            f.write(content)

    def init_server_groups_org(self):
        # Get the original server_group_list information, and the sub-node carries the IP.
        group_dict = {}
        start_cluster_id = 0
        for sever_group in self._rank_table_info["server_group_list"]:
            group_id = sever_group.get("group_id")
            group_type = self.get_server_role()
            if group_type is None:
                raise ValueError("Unknown group id.")
            group = ServerGroup(sever_group, start_cluster_id=start_cluster_id)
            group_dict.setdefault(group_id, group)
            start_cluster_id += sum(len(server.device_list) for server in group.server_list)
        return group_dict

    def sort_server_list(self, server_list):
        ip_list = []
        for unit in self.global_decode_server_ip_list:
            if server_list[0].get("server_ip"):
                ip_list = unit

        server_list_tmp = []
        rank_id = 0
        for ip in ip_list:
            for server_info in server_list:
                if server_info.get("server_ip") == ip:
                    device_list = []
                    server_info_tmp = deepcopy(server_info)
                    for device in server_info.get("device"):
                        device_tmp = deepcopy(device)
                        device_tmp["rank_id"] = str(rank_id)
                        rank_id += 1
                        device_list.append(device_tmp)
                    server_info_tmp["device"] = device_list
                    server_list_tmp.append(server_info_tmp)
                    continue
        return server_list_tmp

    def init_server_groups(self):
        # Obtain the original server_group_list information, merge device, and the sub-node not carries the IP.
        group_dict = {}
        start_cluster_id = 0
        for server_group in self._rank_table_info["server_group_list"]:
            group_id = server_group.get("group_id")
            group_type = self.get_server_role()
            if group_type is None:
                raise ValueError("Unknown group id.")

            server_group_tmp = deepcopy(server_group)

            sorted_ip = server_group.get("server_list")
            if int(group_id) >= int(self.prefill_pod_num):
                sorted_ip = self.sort_server_list(server_group.get("server_list"))

            device_list = []
            for server_info in sorted_ip:
                for device in server_info.get("device"):
                    device_list.append(device)

            server_list_tmp = deepcopy(server_group.get("server_list")[0])
            server_list_tmp["device"] = device_list
            server_group_tmp["server_list"] = [server_list_tmp]

            group = ServerGroup(server_group_tmp, start_cluster_id=start_cluster_id)
            group_dict.setdefault(group_id, group)
            start_cluster_id += sum(len(server.device_list) for server in group.server_list)
        return group_dict

    def get_group_type_from_server(self, server: Server):
        for group_type, group in self.group_dict.items():
            if group.contains(server):
                return group_type
        raise ValueError(f"Server ip {server.server_ip} is not in the groups.")

    def get_server_role(self):
        role = os.environ.get("ROLE")
        return role


    def get_cluster_id(self, server: Server):
        for _, group in self.group_dict.items():
            if group.contains(server):
                return group.get_cluster_id(server)
        return 0

    def get_instance_num(self):
        return self.prefill_pod_num + self.decode_pod_num

    @property
    def scheduler_group(self):
        return self.group_dict.get(GroupType.SCHEDULER, None)

    @property
    def prefill_group(self):
        group_id_start = 0
        group_id_end = self.prefill_pod_num
        prefill_group_list = [self.group_dict_org.get(str(i), None) for i in range(group_id_start, group_id_end)]
        return prefill_group_list

    @property
    def prefill_group_server_list(self):
        prefill_server_list_all = []
        for i in range(0, int(self.prefill_pod_num)):

            server_list_tmp = deepcopy(self.prefill_group[i].server_list[0])
            device_list = []
            for server_info in self.prefill_group[i].server_list:
                for device in server_info.device_list:
                    device_list.append(device)

            server_list_tmp.device_list = device_list
            prefill_server_list_all.append(server_list_tmp)
        return prefill_server_list_all

    @property
    def decode_group(self):
        group_id_start = self.prefill_pod_num
        group_id_end = self.prefill_pod_num + self.decode_pod_num
        decode_group_list = [self.group_dict_org.get(str(i), None) for i in range(group_id_start, group_id_end)]
        return decode_group_list

    @property
    def decode_group_server_list(self):
        # Check the correct position of the user input IP in the rank table
        decode_server_list_all = []

        for decode_pod_ip in self.global_decode_server_ip_list:
            # Each D instance generates a Server class, and the devices are sorted by the IP address of decode_pod.
            device_list_tmp = []
            rank_id = 0
            server_tmp = deepcopy(self.decode_group[0].server_list[0])
            for ip in decode_pod_ip:
                success = False
                for group_info in self.decode_group:
                    server_list = group_info.server_list
                    for server in server_list:
                        if server.server_ip == ip:
                            for device in server.device_list:
                                device_info = {
                                    "device_id": device.device_id,
                                    "device_ip": device.device_ip,
                                    "rank_id": str(rank_id)
                                }

                                rank_id += 1
                                device_list_tmp.append(Device(device_info))
                        if server.server_ip == decode_pod_ip[0]:
                            server_tmp = deepcopy(server)
                        success = True
                        continue
                if not success:
                    raise ValueError(
                        f"({ip}) NOT in gloal_ranktable, please check the DECODE_SERVER_IP_LIST env)")

            server_tmp.device_list = device_list_tmp
            decode_server_list_all.append(server_tmp)
        return decode_server_list_all

