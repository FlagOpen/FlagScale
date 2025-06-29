# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

class Device:
    def __init__(self, device_info):
        self.device_id = int(device_info["device_id"])
        self.device_ip = device_info["device_ip"]
        self.rank_id = int(device_info["rank_id"])

    def __repr__(self) -> str:
        return ("Device("
                f"device_id={self.device_id}, "
                f"device_ip={self.device_ip}, "
                f"rank_id={self.rank_id}")

    def __eq__(self, other):
        return self.device_ip == other.device_ip

    def get_numa_config_format(self):
        return dict(item_id=self.rank_id, device_id=self.device_id, ipaddr=self.device_ip)


class Server:
    def __init__(self, server_info, cluster_id=0):
        self.server_id = server_info["server_id"]
        self.server_ip = server_info["server_ip"]
        self.cluster_id = cluster_id
        self.device_list = self.init_device_list(server_info)

    @staticmethod
    def init_device_list(server_info):
        device_list = []
        for device_info in server_info.get("device", []):
            device_list.append(Device(device_info))
        return device_list

    def __len__(self):
        return len(self.device_list)

    def __repr__(self) -> str:
        return ("Server("
                f"server_id={self.server_id}, "
                f"server_ip={self.server_ip}, "
                f"device_list={self.device_list}")

    def __eq__(self, other):
        return self.server_ip == other.server_ip and self.device_list == other.device_list

    def get_numa_config_format(self):
        return [device.get_numa_config_format() for device in self.device_list]

    def get_server_ip(self):
        return self.server_ip


class ServerGroup:
    def __init__(self, group_info, start_cluster_id=0):
        self.group_id = group_info["group_id"]
        self.server_count = int(group_info["server_count"])
        self.server_list = self.init_server_list(group_info, start_cluster_id)

    @staticmethod
    def init_server_list(group_info, start_cluster_id):
        server_list = []
        for cluster_id, server_info in enumerate(group_info["server_list"]):
            server_list.append(
                Server(server_info, start_cluster_id + cluster_id))
        return server_list

    def __repr__(self) -> str:
        return ("Group("
                f"group_id={self.group_id}, "
                f"server_count={self.server_count}, "
                f"server_list={self.server_list}")

    def get_server_list_ip(self):
        return [server.server_ip for server in self.server_list]

    def contains(self, server: Server):
        for this_server in self.server_list:
            if this_server == server:
                return True
        return False

    def get_cluster_id(self, server: Server):
        for this_server in self.server_list:
            if this_server == server:
                return this_server.cluster_id
        raise ValueError("can not find the server in this group")
