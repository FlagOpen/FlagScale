# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import argparse
import concurrent.futures
import json
import logging
import os
import re
import random
import threading
import time
import weakref
import gc

import func_timeout.exceptions
from func_timeout import func_set_timeout

gc.disable()

import numpy as np
import requests
import yaml

from benchmark_utils import LATENCY_RESERVATION_BITS, THROUGHPUT_RESERVATION_BITS, TP90, TP95, TP99

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(lineno) - %(message)s')
logger = logging.getLogger(__name__)
IS_DEBUG = int(os.environ.get("BENCHMARK_DEBUG", 0))
CHUNK_SIZE = 1024
CHUNK_LINE_START_PREFIX = "data: "
CHUNK_LINE_END_TAG = "[DONE]"
# 在启动线程池前，程序存在两个线程：主线程 和 tqdm
INITIAL_THREADS = 2


class NoReuseThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    # 该函数继承自concurrent/futures/thread.py的ThreadPoolExecutor
    # _adjust_thread_count 重写是为了规避线程复用导致的并发量爬坡时偏小的问题
    def _adjust_thread_count(self):
        # When the executor gets lost, the weakref callback will wake up
        # the worker threads.
        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = '%s_%d' % (self._thread_name_prefix or self,
                                     num_threads)
            t = threading.Thread(name=thread_name, target=concurrent.futures.thread._worker,
                                 args=(weakref.ref(self, weakref_cb),
                                       self._work_queue,
                                       self._initializer,
                                       self._initargs))
            t.daemon = True
            t.start()
            self._threads.add(t)
            concurrent.futures.thread._threads_queues[t] = self._work_queue


class RequestHandler:
    def __init__(self, provider, parallel_num, output_length, test_data):
        self.provider_name = provider.get("name")
        self.api_key = provider.get("api_key")
        # 兼容base_url最后带有/的场景
        base_url = provider.get("base_url")
        if base_url.endswith('/'):
            base_url = base_url.rstrip('/')
        self.base_url = base_url
        self.model_name = provider.get("model_name")
        self.model_category = provider.get("model_category")

        self.parallel_num = parallel_num
        self.output_length = output_length
        self.test_data = test_data

        # 构建client
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=self.parallel_num, pool_maxsize=self.parallel_num)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        self.processing_count = 0
        self.lock = threading.Lock()

    def add_processing_count(self):
        with self.lock:
            self.processing_count += 1

    def reduce_processing_count(self):
        with self.lock:
            self.processing_count -= 1

    def get_rest_processing_count(self):
        with self.lock:
            rest_count = self.parallel_num - self.processing_count
            return rest_count, self.parallel_num, self.processing_count

    def generate_result_template(self):
        result = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "content_tokens": 0,
            "total_tokens": 0,
            "reasoning_piece": None,
            "content_piece": None,
            "reasoning_start_time": None,
            "reasoning_end_time": None,
            "content_start_time": None,
            "content_end_time": None,
            "start_time": time.time(),
            "end_time": None,
            "time_to_first_token": None,
            "first_decode_token_start_time": None,
            "second_decode_token_start_time": None,
            "second_decode_completion_tokens": 2,
            "time_per_output_token": None,
            "time_per_output_token_from_second_decode_token": None,
            "time_between_first_and_second_token": None,
            "reasoning_time": None,
            "content_time": None,
            "total_time": None,
            "total_decode_time": None
        }
        return result

    def refine_data(self, sample):
        pattern = r"\{\"content\":\"(.*?)\"\}"
        matched_results = re.findall(pattern, sample)
        if len(matched_results) > 0:
            content = matched_results[0]
            if len(content) > 0:
                result_a = '{"content":"' + content + '"}'
                result_b = '{"content":"placeholder"}'
                sample_new = sample.replace(result_a, result_b)
            else:
                sample_new = sample
        else:
            sample_new = sample.replace("'", '"')
        return sample_new

    def parse_chunk_data(self, chunk, result):
        chunk_str = chunk.decode('utf-8', 'ignore')
        lines = chunk_str.split('\n')
        for line in lines:
            if line.startswith(CHUNK_LINE_START_PREFIX):
                cur_time = time.time()
                # 最后的一个chunk是 data: [Done], 没有choices
                if line.endswith(CHUNK_LINE_END_TAG):
                    continue
                # 解析 JSON 数据
                json_data = json.loads(line[len(CHUNK_LINE_START_PREFIX):])
                if "choices" not in json_data:
                    continue

                if 'usage' not in json_data or len(json_data['usage']) <= 0:
                    raise ValueError("chunk data does not contain usage data.")
                usage = json_data['usage']
                result["completion_tokens"] = usage['completion_tokens']
                result["prompt_tokens"] = usage['prompt_tokens']
                result["total_tokens"] = usage['total_tokens']

                if result["time_to_first_token"] is None:
                    first_token_end_time = cur_time
                    result["time_to_first_token"] = first_token_end_time - result["start_time"]
                    result["first_decode_token_start_time"] = first_token_end_time
                elif (result["first_decode_token_start_time"] is not None) and (
                        result["second_decode_token_start_time"] is None):
                    result["second_decode_token_start_time"] = cur_time
                    result["second_decode_completion_tokens"] = result["completion_tokens"]

    @func_set_timeout(90)
    def run_prefill(self, response_generator, result):
        try:
            for chunk in response_generator:
                if chunk:
                    self.parse_chunk_data(chunk, result)
                if result["time_to_first_token"] is not None:
                    return "Success"
        except Exception as e:
            logger.error(f"Thread name: {threading.current_thread().name}, provider name: {self.provider_name},"
                         f"error message: {e} chunk {chunk}")
            return "Failed"


    @func_set_timeout(600)
    def model_running(self, response, result):
        response.raise_for_status()
        response_generator = response.iter_content(chunk_size=CHUNK_SIZE)

        try:
            prefill_phase = self.run_prefill(response_generator, result)
            if prefill_phase == "Failed":
                return "Failed"
        except (func_timeout.exceptions.FunctionTimedOut, Exception) as e:
            logger.error(
                f"run_prefill Thread name: {threading.current_thread().name}, provider name: {self.provider_name},"
                f"error message: {e}")
            return "Failed"

        try:
            for chunk in response_generator:
                if chunk:
                    self.parse_chunk_data(chunk, result)
            result["end_time"] = time.time()
            result["total_time"] = result["end_time"] - result["start_time"]

            result["time_between_first_and_second_token"] = result["second_decode_token_start_time"] - result[
                "first_decode_token_start_time"]
            result["decode_total_time_from_first_decode_token"] = result["end_time"] - result[
                "first_decode_token_start_time"]
            result["decode_total_time_from_second_decode_token"] = result["end_time"] - result[
                "second_decode_token_start_time"]
            result["time_per_output_token"] = (result["decode_total_time_from_first_decode_token"] /
                                               (result["completion_tokens"] -
                                                1)) if (result["completion_tokens"] > 1) else 0

            result["time_per_output_token_from_second_decode_token"] = (
                        result["decode_total_time_from_second_decode_token"] /
                        (result["completion_tokens"] - result["second_decode_completion_tokens"])) \
                if (result["completion_tokens"] > result["second_decode_completion_tokens"]) else 0
            result["reasoning_time"] = (result["reasoning_end_time"] - result["reasoning_start_time"]) if (
                    result["reasoning_start_time"] and result["reasoning_end_time"]) else 0
            result["content_time"] = (result["content_end_time"] - result["content_start_time"]) if (
                        result["content_start_time"] and result["content_end_time"]) else 0
            return "Success"
        except Exception as e:
            logger.error(
                f"run decode Thread name: {threading.current_thread().name}, provider name: {self.provider_name},"
                f"error message: {e}")
            return "Failed"

    def _send_request(self, is_warmup=False):
        sample = random.choice(self.test_data)
        result = self.generate_result_template()

        prompt = sample["input"]
        if is_warmup:
            prompt = prompt[-256:]
        output_length = sample.get("output_tokens", self.output_length)

        if self.api_key:
            headers = {
                'Content-Type': 'application/json'
            }
        else:
            headers = {'Content-Type': 'application/json'}

        data = {
            'model': self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            'temperature': 0.6,
            'max_tokens': output_length,
            'stream': True,
            'stream_options': {"include_usage": True, "continuous_usage_stats": True},
            'repetition_penalty': 1.0,
            'ignore_eos': True
        }
        try:
            response = self.session.post(
                self.base_url,
                headers=headers,
                json=data,
                stream=True,
                verify=False,
                timeout=None
            )
        except Exception as e:
            logger.error(f"=========== err:{e}")
        finally:
            self.session.close()
        try:
            exec_phase = self.model_running(response, result)
        except (func_timeout.exceptions.FunctionTimedOut, Exception) as e:
            logger.error(f"Thread name: {threading.current_thread().name}, provider name: {self.provider_name},"
                         f"error message: {e}")
            exec_phase = "Failed"
        self.reduce_processing_count()
        response.close()

        if exec_phase == "Failed":
            return exec_phase, result
        else:
            final_result = {
                "start_time": result["start_time"],
                "end_time": result["end_time"],
                "prompt_tokens": result["prompt_tokens"],
                "reasoning_tokens": result["reasoning_tokens"],
                "content_tokens": result["content_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "time_to_first_token": result["time_to_first_token"],
                "time_per_output_token": result["time_per_output_token"],
                "time_per_output_token_from_second_decode_token": result[
                    "time_per_output_token_from_second_decode_token"],
                "time_between_first_and_second_token": result["time_between_first_and_second_token"],
                "reasoning_time": result["reasoning_time"],
                "content_time": result["content_time"],
                "total_time": result["total_time"],
                "total_decode_time": result["total_time"] - result["time_to_first_token"]
            }
            return exec_phase, final_result


def calculate_metrics(results, request_num, total_time):
    # 计算输出的总 token 数
    all_completion_tokens = [result['completion_tokens'] for result in results]
    tp90_completion_tokens = np.percentile(all_completion_tokens, TP90)
    tp95_completion_tokens = np.percentile(all_completion_tokens, TP95)
    tp99_completion_tokens = np.percentile(all_completion_tokens, TP99)
    min_completion_tokens = np.min(all_completion_tokens)
    avg_completion_tokens = np.mean(all_completion_tokens)
    total_output_tokens = sum(all_completion_tokens)

    # 计算avg reasoning content length
    all_prompt_tokens = [result['prompt_tokens'] for result in results]
    all_reasoning_tokens = [result['reasoning_tokens'] for result in results]
    all_content_tokens = [result['content_tokens'] for result in results]
    avg_prompt_tokens = np.mean(all_prompt_tokens)
    avg_reasoning_tokens = np.mean(all_reasoning_tokens)
    avg_content_tokens = np.mean(all_content_tokens)

    # 计算 TPS (token per second)
    tps = total_output_tokens / total_time if total_time else 0
    # total token per second
    total_tokens_list = [result['total_tokens'] for result in results]
    ttps = sum(total_tokens_list) / total_time if total_time else 0

    # 计算 TTFT
    ttft_list = [result['time_to_first_token'] for result in results]
    tp90_ttft = np.percentile(ttft_list, TP90)
    tp95_ttft = np.percentile(ttft_list, TP95)
    tp99_ttft = np.percentile(ttft_list, TP99)
    max_ttft = np.max(ttft_list)
    avg_ttft = np.mean(ttft_list)

    # 计算接收率
    # content_tokens: 执行的轮次
    # completion_tokens: 总输出的token数
    accept_rate_list = [(result["completion_tokens"] - 1) / (result["content_tokens"] - 1) - 1 for result in results]
    tp90_accept_rate = np.percentile(accept_rate_list, TP90)
    tp95_accept_rate = np.percentile(accept_rate_list, TP95)
    tp99_accept_rate = np.percentile(accept_rate_list, TP99)
    max_accept_rate = np.max(accept_rate_list)
    avg_accept_rate = np.mean(accept_rate_list)

    # queries_per_second
    queries_per_second = request_num / total_time
    # failure_times
    success_times = len(results)
    failure_times = request_num - success_times
    fail_rate = failure_times / request_num

    # 计算 TPOT from first decode time
    tpot_list = [result['time_per_output_token'] for result in results]
    tp90_tpot = np.percentile(tpot_list, TP90)
    tp95_tpot = np.percentile(tpot_list, TP95)
    tp99_tpot = np.percentile(tpot_list, TP99)
    max_tpot = np.max(tpot_list)
    avg_tpot = np.mean(tpot_list)

    # 计算 TPOT from second decode time
    tpots_second = [result['time_per_output_token_from_second_decode_token'] for result in results]
    tp90_tpot_second = np.percentile(tpots_second, TP90)
    tp95_tpot_second = np.percentile(tpots_second, TP95)
    tp99_tpot_second = np.percentile(tpots_second, TP99)
    max_tpot_second = np.max(tpots_second)
    avg_tpot_second = np.mean(tpots_second)

    # 计算首token和第一个增量token之间的时间
    time_between_first_and_second_token = [result['time_between_first_and_second_token'] for result in results]
    tp90_time_between_first_and_second_token = np.percentile(time_between_first_and_second_token, TP90)
    tp95_time_between_first_and_second_token = np.percentile(time_between_first_and_second_token, TP95)
    tp99_time_between_first_and_second_token = np.percentile(time_between_first_and_second_token, TP99)
    min_time_between_first_and_second_token = np.min(time_between_first_and_second_token)
    max_time_between_first_and_second_token = np.max(time_between_first_and_second_token)
    avg_time_between_first_and_second_token = np.mean(time_between_first_and_second_token)

    # 计算e2e
    e2e_list = [result['total_time'] for result in results]
    tp90_e2e = np.percentile(e2e_list, TP90)
    tp95_e2e = np.percentile(e2e_list, TP95)
    tp99_e2e = np.percentile(e2e_list, TP99)
    max_e2e = np.max(e2e_list)
    avg_e2e = np.mean(e2e_list)

    logger.info(f"\n")
    logger.info(f"TP90_COMPLETION_TOKENS: {tp90_completion_tokens:.2f} tokens")
    logger.info(f"TP95_COMPLETION_TOKENS: {tp95_completion_tokens:.2f} tokens")
    logger.info(f"TP99_COMPLETION_TOKENS: {tp99_completion_tokens:.2f} tokens")
    logger.info(f"MIN_COMPLETION_TOKENS: {min_completion_tokens} tokens")
    logger.info(f"AVG_COMPLETION_TOKENS: {avg_completion_tokens:.2f} tokens")
    logger.info(f"AVG_REASONING_TOKENS: {avg_reasoning_tokens:.2f} tokens")
    logger.info(f"AVG_CONTENT_TOKENS: {avg_content_tokens:.2f} tokens")
    logger.info(f"AVG_PROMPT_TOKENS: {avg_prompt_tokens:.2f} tokens")
    logger.info(f"Output_Token_Throughput: {tps:.2f} tokens/s")
    logger.info(f"Total_Token_Throughput: {ttps:.2f} tokens/s")
    logger.info(f"TP90_TTFT: {tp90_ttft:.3f} s")
    logger.info(f"TP95_TTFT: {tp95_ttft:.3f} s")
    logger.info(f"TP99_TTFT: {tp99_ttft:.3f} s")
    logger.info(f"MAX_TTFT: {max_ttft:.4f} s")
    logger.info(f"AVG_TTFT: {avg_ttft:.4f} s")
    logger.info(f"TP90_TPOT from first token: {tp90_tpot:.3f} s")
    logger.info(f"TP95_TPOT from first token: {tp95_tpot:.3f} s")
    logger.info(f"TP99_TPOT from first token: {tp99_tpot:.3f} s")
    logger.info(f"MAX_TPOT from first token: {max_tpot:.4f} s")
    logger.info(f"AVG_TPOT from first token: {avg_tpot:.4f} s")
    logger.info(f"TP90_TPOT from second token: {tp90_tpot_second:.3f} s")
    logger.info(f"TP95_TPOT from second token: {tp95_tpot_second:.3f} s")
    logger.info(f"TP99_TPOT from second token: {tp99_tpot_second:.3f} s")
    logger.info(f"MAX_TPOT from second token: {max_tpot_second:.4f} s")
    logger.info(f"AVG_TPOT from second token: {avg_tpot_second:.4f} s")

    logger.info(f"TP90_accept_rate: {tp90_accept_rate:.3f}")
    logger.info(f"TP95_accept_rate: {tp95_accept_rate:.3f}")
    logger.info(f"TP99_accept_rate: {tp99_accept_rate:.3f}")
    logger.info(f"MAX_accept_rate: {max_accept_rate:.3f}")
    logger.info(f"AVG_accept_rate: {avg_accept_rate:.3f}")

    logger.info(f"TP90 time between first and second token: {tp90_time_between_first_and_second_token:.3f} s")
    logger.info(f"TP95 time between first and second token: {tp95_time_between_first_and_second_token:.3f} s")
    logger.info(f"TP99 time between first and second token: {tp99_time_between_first_and_second_token:.3f} s")
    logger.info(f"Min time between first and second token: {min_time_between_first_and_second_token:.3f} s")
    logger.info(f"Max time between first and second token: {max_time_between_first_and_second_token:.3f} s")
    logger.info(f"AVG time between first and second token: {avg_time_between_first_and_second_token:.3f} s")
    logger.info(f"TP90_E2E: {tp90_e2e:.3f} s")
    logger.info(f"TP95_E2E: {tp95_e2e:.3f} s")
    logger.info(f"TP99_E2E: {tp99_e2e:.3f} s")
    logger.info(f"MAX_E2E: {max_e2e:.3f} s")
    logger.info(f"AVG_E2E: {avg_e2e:.3f} s")
    logger.info(f"QPS: {queries_per_second:.2f}")
    logger.info(f"Total_Time: {total_time:.3f} s")
    logger.info(f"Failure_Times: {failure_times}")
    logger.info(f"Total_Times: {request_num}")
    logger.info(f"Fail_Rate: {fail_rate:.4f}")
    logger.info(f"---------------------------\n")


def test_concurrent_performance(provider, dataset_dir, parallel_num=512, input_length=256,
                                growth_rate=0, rounds=2, output_length=256):

    with open(f'{dataset_dir}/{input_length}.json', "r", encoding="utf-8") as f:
        test_data = json.load(f)
    handler = RequestHandler(provider, parallel_num, output_length, test_data)

    with NoReuseThreadPoolExecutor() as executor:
        executor._max_workers = parallel_num

        # warm_up阶段，先一把发送parallel_num // 2个请求，然后按照一定growth_rate增加
        warmup_futures = []

        # 爬坡阶段
        logger.info("=============climbing period")
        # for warm_index in range(parallel_num * 2 // growth_rate):
        warm_index = 0
        while abs(handler.get_rest_processing_count()[0] - parallel_num) > 10:
            for _ in range(growth_rate):
                while True:
                    cur_reqs, _, _ = handler.get_rest_processing_count()
                    if cur_reqs > 0:
                        break
                    time.sleep(0.01)
                handler.add_processing_count()
                future = executor.submit(handler._send_request, True)
                warmup_futures.append(future)
            time.sleep(1)
            warm_index += 1
            if warm_index % 10 == 0:
                logger.info("======warmup_index: {}".format(warm_index))

        # 稳态阶段
        logger.info("=============stable period")
        futures = []
        zero_count = 0
        total_reqs = rounds * parallel_num
        while True:
            rest_reqs = total_reqs - len(futures)
            if rest_reqs <= 0:
                break
            cur_reqs, cur_parallel_num, cur_process_count = handler.get_rest_processing_count()
            for _ in range(min(cur_reqs, rest_reqs)):
                handler.add_processing_count()
                future = executor.submit(handler._send_request)
                futures.append(future)
            logger.info(
                "=========total_reqs: {}  cur_reqs: {} cur_parallel_num: {} cur_process_count: {}".format(len(futures),
                cur_reqs, cur_parallel_num, cur_process_count))
            if cur_reqs == 0:
                zero_count += 1
            if zero_count >= 100:
                break
            time.sleep(1)

    # 等待所有任务完成
    logger.info("=====All requests are submitted, waiting for finished")
    concurrent.futures.wait(warmup_futures)
    concurrent.futures.wait(futures)

    results = [future.result()[1] for future in futures if future.result()[0] == "Success"]

    with open("hjhjhj_tx.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    end_time_list = [result["end_time"] for result in results]
    min_end_time = min(end_time_list)
    max_end_time = max(end_time_list)
    time_interval = max_end_time - min_end_time

    start_time = min_end_time + time_interval / 4
    end_time = start_time + time_interval / 2

    results_middle = []
    for result in results:
        if start_time <= result["end_time"] <= end_time:
            results_middle.append(result)

    request_num = len(results_middle)
    with open("hjhjhj_tx1.json", "w") as f:
        json.dump(results_middle, f, indent=4, ensure_ascii=False)
    logger.info("====all request finished!!!!")
    handler.session.close()
    calculate_metrics(results_middle, request_num, time_interval / 2)


def run_climbing(args: argparse.Namespace):
    # 读取 providers 配置
    providers_path = os.path.realpath(args.providers_path)
    with open(providers_path, "r", encoding="utf-8") as f:
        providers = yaml.safe_load(f)
    # 过滤 api_key 和 base_url 不同时为空的服务提供商(provider)
    providers = [provider for provider in providers["providers"]
                 if (len(provider["api_key"]) > 0
                     and len(provider["base_url"]) > 0
                     and len(provider["model_name"]) > 0)
                 ]
    if len(providers) == 0:
        raise ValueError("There must be at least one valid provider, "
                         "A valid provider must have api_key, base_url and model_name.")

    dataset_dir = args.dataset_dir
    prompt_tokens = args.prompt_tokens
    output_tokens = args.output_tokens
    rounds = args.rounds
    growth_rate = args.growth_rate
    parallel_num = args.parallel_num

    # 循环对每个服务商进行测试
    for provider in providers:
        provider_name = provider.get("name")
        model_category = provider.get("model_category")

        logger.info(f"\n---------------------------")
        logger.info(f"开始测试服务商：{provider_name}")
        logger.info(f"模型类型：{model_category}")
        logger.info(f"并发数： {parallel_num}")
        logger.info(f"轮数： {rounds}")

        test_concurrent_performance(provider, dataset_dir, parallel_num, prompt_tokens, growth_rate, rounds,
                                    output_tokens)


def main(args: argparse.Namespace):
    run_climbing(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the DeepSeek latency of processing a single batch of requests till completion.")
    parser.add_argument("--providers-path", type=str, default="./providers.yaml",
                        help="providers configuration file path")
    parser.add_argument("--dataset-dir", type=str, default="./", help="test dataset directory")
    parser.add_argument("--parallel-num", type=int, default=1536, help="Parallel Number")
    parser.add_argument("--prompt-tokens", type=int, default=300028192, help="prompt tokens")
    parser.add_argument("--output-tokens", type=int, default=1024, help="Max output tokens")
    parser.add_argument("--rounds", type=int, default=10, help="the round of batch")
    parser.add_argument("--growth-rate", type=int, default=48)
    args = parser.parse_args()
    main(args)

