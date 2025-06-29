# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import math


#  | ------------------------------- prefill -------------------------------------|
#  | ------------- prefil0 ------------- |  | ------------ prefill0` ------------ | # copy means p id is same
#  |tp| |tp| |tp| |tp| |tp| |tp| |tp| |tp|  |tp| |tp| |tp| |tp| |tp| |tp| |tp| |tp| 
# 
#   decode0   decode1   decode0   decode0    decode0   decode1   decode0   decode1
#   d0t0 d0t1 d0t0 d0t1 d1t0 d1t1 d1t0 d1t1  d2t0 d2t1 d2t0 d2t1 d3t0 d3t1 d3t0 d3t1
#  | ------------------------------- decode- -------------------------------------|

#  | ------------------------------- prefill -------------------------------------|
#  | ------------- prefil0 ------------- |  | ------------ prefill1 ------------- | 
#  |tp| |tp| |tp| |tp| |tp| |tp| |tp| |tp|  |tp| |tp| |tp| |tp| |tp| |tp| |tp| |tp| 
# 
#   decode0   decode1   decode0   decode0    decode0   decode1   decode0   decode1
#   d0t0 d0t1 d0t0 d0t1 d1t0 d1t1 d1t0 d1t1  d0t0 d0t1 d0t0 d0t1 d1t0 d1t1 d1t0 d1t1 # copy means d id is same
#  | ------------------------------- decode- -------------------------------------|
def get_p_start_rank(p_tp_size, p_dp_size, d_tp_size, d_dp_size, d_node_num, cur_d_node, cur_d_dp):
    p_rank_num = p_tp_size * p_dp_size

    # When D builds a link to P, it needs to hash the device connected to P.
    # Prefill is segmented by decode node * Decode DP size, and each Decode DP is connected to a different segment.
    link_rank_step = math.ceil(p_tp_size / (d_dp_size * d_node_num))

    return ((cur_d_node + cur_d_dp * d_node_num) * link_rank_step) % p_rank_num


def prepare_ranktables(prefill_node, decode_node, p_rank_start, p_rank_end, d_rank_start, d_rank_end, p_start_rank):
    p_ranktables, d_ranktables = {}, {}
    p_tp, d_tp = p_rank_end - p_rank_start, d_rank_end - d_rank_start
    if d_tp > p_tp:
        raise ValueError("decode tp must <= prefill tp size currently")

    for d_ranks in range(d_rank_start, d_rank_end):
        # The connected prefill rank is added to the external count p_start_rank, and random processing is removed.
        p_ranks = p_start_rank % p_tp
        p_start_rank += 1
        p_device = prefill_node.device_list[p_ranks]
        d_device = decode_node.device_list[d_ranks]
        rank_table_dict = {
            "server_count": "1",
            "status": "completed",
            "version": "1.0",
            "server_list": [
            ]
        }

        if prefill_node.server_ip != decode_node.server_ip:  # on multiple machines
            p_devices = {
                "device": [
                    {
                        "device_id": str(p_device.device_id),
                        "device_ip": p_device.device_ip,
                        "rank_id": "0"
                    }
                ],
                "server_id": prefill_node.server_ip
            }
            d_devices = {
                "device": [
                    {
                        "device_id": str(d_device.device_id),
                        "device_ip": d_device.device_ip,
                        "rank_id": "1"
                    }
                ],
                "server_id": decode_node.server_ip
            }
            rank_table_dict["server_list"].append(p_devices)
            rank_table_dict["server_list"].append(d_devices)
            rank_table_dict["server_count"] = "2"  # used to be 2

        else:  # on single machine
            pd_devices = {
                "device": [
                    {
                        "device_id": str(p_device.device_id),
                        "device_ip": p_device.device_ip,
                        "rank_id": "0"
                    },
                    {
                        "device_id": str(d_device.device_id),
                        "device_ip": d_device.device_ip,
                        "rank_id": "1"  # used to be 1
                    }
                ],
                "server_id": prefill_node.server_ip
            }
            rank_table_dict["server_list"].append(pd_devices)
            rank_table_dict["server_count"] = "1"

        p_ranktables[p_ranks] = rank_table_dict
        d_ranktables[d_ranks] = rank_table_dict
    return p_ranktables, d_ranktables
