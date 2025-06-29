# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import logging
import argparse
import torch
import torch_npu
import Levenshtein
import pytest
from vllm import LLM, SamplingParams
from tests.mark_utils import arg_mark


def get_new_response():
    import os
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["REPLAY_MODE"] = "REPLAY"  # replay inputs and outputs from the cache
    os.environ["MOCK_CAPTURE_DIR"] = "/home/ma-user/modelarts/inputs/data_url_0/capture/"
    os.environ["MOCK_CAPTURE_FILE"] = "/home/ma-user/modelarts/inputs/data_url_0/capture/.mock_qwen2_5"
    os.environ["MOCK_CAPTURE_FILE_LOCK"] = "/home/ma-user/modelarts/inputs/data_url_0/capture/.lock"

    prompts = [
                  "最近在看《飞鸟集》，这本书首次初版于哪一年？",
              ] * 1
    import random
    random.shuffle(prompts)
    sampling_params = SamplingParams(max_tokens=100, temperature=0.0, top_p=0.95)
    llm = LLM(model="/home/ma-user/modelarts/inputs/data_url_0/model/Qwen2.5-0.5B-Instruct",
              tensor_parallel_size=2,
              trust_remote_code=True,
              enforce_eager=True,
              max_model_len=1024,
              gpu_memory_utilization=0.9)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        logging.info(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        return generated_text
    return None


def compare(args):
    source_logs = [{"response": '（ ） A. 1925年 B. 1926年 C. 1927年 D. 1928年\n1926年\n\n《飞鸟集》的作者是（ ）。 A. 鲁迅 B. 老舍 C. 茅盾 D. '
                                '郁达夫\n郁达夫\n\n《飞鸟集》的作者是（ ）。 A. 鲁迅 B'}]
    target_logs = [{"response": ''}]
    target_logs[0]["response"] = get_new_response()
    max_length = min(len(target_logs), len(source_logs))
    target_logs = target_logs[:max_length]
    source_logs = source_logs[:max_length]

    distance_sum = 0.
    distance_percent = 0.
    source_length = 0.
    target_length = 0.
    for batch_idx in range(max_length):
        source_item = source_logs[batch_idx]
        target_item = target_logs[batch_idx]
        source_str = source_item['response'][:args.sequence_length]
        target_str = target_item['response'][:args.sequence_length]
        distance = Levenshtein.distance(source_str, target_str)
        distance_sum += distance
        distance_percent += distance / (len(source_str) + 1e-6)
        source_length += len(source_item['response'])
        target_length += len(target_item['response'])

        if args.debug:
            logging.info("[{}] Source str: {}".format(batch_idx, source_str[:args.sequence_length]))
            logging.info("[{}] Target str: {}".format(batch_idx, target_str[:args.sequence_length]))

    target_length /= max_length
    source_length /= max_length
    distance_sum /= max_length
    distance_percent /= max_length
    logging.info("edit distance: %.2f, sequence length: %d, edit distance percent: %.2f%%." \
                 " source length: %d, target length: %d" % (
                     distance_sum, args.sequence_length, distance_percent * 100, source_length, target_length))
    return distance_sum


@arg_mark(['platform_ascend910b'], 'level1')
def test_edit_distance(**kwargs):
    # 设置默认参数
    default_args = {
        'sequence_length': 40,
        'max_distance': 0,
        'filter_batch_idx': None,
        'debug': False
    }
    # 更新默认参数与传入的kwargs
    default_args.update(kwargs)

    # 创建命名空间对象
    args = argparse.Namespace(
        **default_args
    )

    distance = compare(args)

    if distance > args.max_distance:
        assert False, f"Edit distance should be within {args.max_distance}/{args.sequence_length}, \
            but got {distance}/{args.sequence_length}."
    else:
        logging.info("Check passed!")


if __name__ == '__main__':
    pytest.main()
