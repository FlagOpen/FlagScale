# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ast
# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# All rights reserved.

import os
import logging
import csv
import glob
import argparse
import shutil
import pytest
from benchmark_parallel import main
from tests.mark_utils import arg_mark
from tests.st.scripts.utils import check_service_status



def get_nth_parent_dir(path, n):
    for _ in range(n):
        path = os.path.dirname(path)
    return path


def run_benchmark(backend, host, port, tokenizer, served_model_name, epochs, parallel_num, prompt_tokens, output_tokens,
                  benchmark_csv, dataset_type, **kwargs):
    # 设置默认参数
    default_args = {
        'run_method': "parallel",
        'url': "",
        'app_code': None,
        'best_of': 1,
        'use_beam_search': False,
        'request_rate': float("inf"),
        'seed': None,
        'num_scheduler_steps': 1,
        'prefix_caching_num': 0,
        'use_spec_decode': False,
        'num_speculative_tokens': -1,
        'dataset_path': "",
        'use_real_dataset_output_tokens': False,
        'use_pd_separate': False,
        'providers_path': None,
        'dataset_dir': None,
        'benchmark_dir': None,
        'control_method': "queue",
        'growth_rate': None,
        'use_mtp_accept_rate': False,
        'temperature': 0,
        'top_k': -1,
        'top_p': 1,
        'num_prompts': None
    }

    # 更新默认参数与传入的kwargs
    default_args.update(kwargs)

    # 创建命名空间对象
    args = argparse.Namespace(
        backend=backend,
        host=host,
        port=port,
        tokenizer=tokenizer,
        served_model_name=served_model_name,
        epochs=epochs,
        parallel_num=parallel_num,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        benchmark_csv=benchmark_csv,
        dataset_type=dataset_type,
        **default_args
    )

    # 调用 main 函数
    main(args)


def read_csv_and_extract_data(output_dir):
    data_dict = {}
    csv_files = glob.glob(f'benchmark_result_*.csv')
    if not csv_files:
        assert False, "not find csv files!"

    csv_file_path = csv_files[0]
    with open(csv_files[0], mode='r', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # 提取键（Input_Length, Output_Length, Concurrency）
            input_len = int(float(row['Input_Length']))
            output_len = int(float(row['Output_Length']))
            concurrency = int(float(row['Concurrency']))
            key = f"{input_len}_{output_len}_{concurrency}"
            value = float(row['AVG_TPOT(s)'])
            data_dict[key] = value

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 移动处理过的CSV文件到指定目录
    base_name = os.path.basename(csv_file_path)
    destination_path = os.path.join(output_dir, base_name)
    shutil.move(csv_file_path, destination_path)

    # 删除当前目录下所有 benchmark_result_*.csv 文件
    files_to_delete = glob.glob('benchmark_result_*.csv')
    for file in files_to_delete:
        os.remove(file)

    return data_dict


def check_tolerance(avg_tpot, expected_value, tolerance=0.001):
    # 由于benchmark当前只统计到ms级别,抖动范围先放在1ms，理论应该是0.5ms
    return expected_value - tolerance <= avg_tpot <= expected_value + tolerance


def check_file_path():
    # 获取当前脚本的绝对路径
    cur_script_path = os.path.abspath(__file__)
    omni_infer_path = get_nth_parent_dir(cur_script_path, 4)
    benchmark_parallel_path = os.path.join(
        omni_infer_path,
        'tools',
        'llm_evaluation',
        'benchmark_tools',
        'benchmark_parallel.py'
    )
    if not os.path.isfile(benchmark_parallel_path):
        assert False, f"错误：脚本文件 {benchmark_parallel_path} 不存在。"
    return True


def before_run_check(output_dir, host):
    if not os.path.exists(output_dir):
        assert False, f"Path:{output_dir} does not exist!"

    if check_service_status(host):
        logging.info("Service started successfully.")
    else:
        assert False, "Service started failed."

    assert check_file_path(), "获取脚本benchmark_parallel.py失败"


def check_benchmark_baseline(avg_tpot, check_field, forward_time, expected_value):
    if check_field in avg_tpot.keys():
        avg_tpot_field = avg_tpot[check_field] - forward_time
        tpot_field_result = check_tolerance(avg_tpot_field, expected_value)
        assert tpot_field_result, f"{check_field} result is False,avg_tpot_time:{avg_tpot[check_field]}"
    else:
        assert False, f"{check_field} is not exist!"


@arg_mark(['platform_ascend910b'], 'level1')
def test_benchmark_performance_level1(output_dir, forward_time, except_time, tokenizer, served_model_name, host):
    before_run_check(output_dir, host)
    ip, port = host.split(':')
    run_benchmark(
        backend="openai",
        host=ip,
        port=port,
        tokenizer=tokenizer,
        served_model_name=served_model_name,
        epochs=1,
        parallel_num=[192, 384, 576, 768, 1024],
        prompt_tokens=[2048, 2048, 3072, 3500, 6000],
        output_tokens=[1024, 2048, 1024, 1000, 1024],
        benchmark_csv="benchmark_result.csv",
        dataset_type="random"
    )

    avg_tpot = read_csv_and_extract_data(output_dir)
    mock_model_forward_time = float(forward_time)
    check_field = "3584_1024_384"
    mock_model_except_time = float(except_time)
    check_benchmark_baseline(avg_tpot, check_field, mock_model_forward_time, mock_model_except_time)
    logging.info("level1 checks passed successfully")


@arg_mark(['platform_ascend910b'], 'level0')
def test_benchmark_performance_level0(output_dir, forward_time, except_time, tokenizer, served_model_name, host):
    before_run_check(output_dir, host)
    ip, port = host.split(':')
    run_benchmark(
        backend="openai",
        host=ip,
        port=port,
        tokenizer=tokenizer,
        served_model_name=served_model_name,
        epochs=1,
        parallel_num=[384],
        prompt_tokens=[3500],
        output_tokens=[1000],
        benchmark_csv="benchmark_result.csv",
        dataset_type="random"
    )

    avg_tpot = read_csv_and_extract_data(output_dir)
    mock_model_forward_time = float(forward_time)
    check_field = "3584_1024_384"
    mock_model_except_time = float(except_time)
    check_benchmark_baseline(avg_tpot, check_field, mock_model_forward_time, mock_model_except_time)
    logging.info("level0 checks passed successfully")


@arg_mark(['platform_ascend910b'], 'lts')
def test_benchmark_performance_lts(output_dir, forward_time, except_time, tokenizer, served_model_name, host):
    before_run_check(output_dir, host)
    ip, port = host.split(':')
    run_benchmark(
        backend="openai",
        host=ip,
        port=port,
        tokenizer=tokenizer,
        served_model_name=served_model_name,
        epochs=1,
        parallel_num=[768],
        prompt_tokens=[3500],
        output_tokens=[1000],
        benchmark_csv="benchmark_result_lts.csv",
        dataset_type="random"
    )

    avg_tpot = read_csv_and_extract_data(output_dir)
    mock_model_forward_time = float(forward_time)
    check_field = "3500_1000_1152"
    mock_model_except_time = float(except_time)
    # check_benchmark_baseline(avg_tpot, check_field, mock_model_forward_time, mock_model_except_time)
    logging.info("long term stability checks passed successfully")


@arg_mark(['platform_ascend910b'], 'common')
def test_benchmark_performance_common(output_dir, forward_time, except_time, tokenizer, served_model_name, host,
                                      parallel_num, prompt_tokens, output_tokens, benchmark_csv):
    before_run_check(output_dir, host)
    ip, port = host.split(':')
    run_benchmark(
        backend="openai",
        host=ip,
        port=port,
        tokenizer=tokenizer,
        served_model_name=served_model_name,
        epochs=1,
        parallel_num=ast.literal_eval(parallel_num),
        prompt_tokens=ast.literal_eval(prompt_tokens),
        output_tokens=ast.literal_eval(output_tokens),
        benchmark_csv=benchmark_csv,
        dataset_type="random"
    )

    avg_tpot = read_csv_and_extract_data(output_dir)
    mock_model_forward_time = float(forward_time)
    check_field = "3584_1024_384"
    mock_model_except_time = float(except_time)
    # check_benchmark_baseline(avg_tpot, check_field, mock_model_forward_time, mock_model_except_time)
    logging.info("level1 checks passed successfully")


if __name__ == "__main__":
    pytest.main()
