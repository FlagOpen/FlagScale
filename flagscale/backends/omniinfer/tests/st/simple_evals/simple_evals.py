# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import argparse
import os
import sys
import time
import traceback

import pandas as pd
import common
import datetime
from drop_eval import DropEval
from gpqa_eval import GPQAEval
from humaneval_eval import HumanEval
from mgsm_eval import MGSMEval
from mmlu_eval import MMLUEval
from sampler.chat_completion_sampler import (
    ChatCompletionSampler,
)


def main(url=None):
    project_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument("--model", type=str, help="Select a model by name", default="deepseek")
    parser.add_argument("--served-model-name", type=str, help="Select a served model name", default="deepseek")
    parser.add_argument("--max-tokens", type=int, help="max tokens", default=2048)
    parser.add_argument("--temperature", type=float, help="temperature", default=0.5)
    parser.add_argument("--num-threads", type=int, help="num threads", default=50)
    parser.add_argument("--url", type=str, help="url", default="http://127.0.0.1:8000/v1/")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--dataset", nargs='+', type=str, help="eval dataset")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    parser.add_argument("--dataset-path", type=str, help="eval dataset path", default=f"{project_path}/dataset")

    if len(sys.argv) <= 2:
        # 代码指定覆盖命令行传递
        args = parser.parse_args([
            "--dataset", "mmlu", "gpqa", "mgsm", "drop", "humaneval",
            "--served-model-name", "deepseek",
            "--url", url if url else "http://127.0.0.1:8000/v1/",
            "--max-tokens", "2048",
            "--temperature", "0.5",
            "--num-threads", "50",
            "--debug"
        ])
    else:
        args = parser.parse_args()

    models = {
        "deepseek": ChatCompletionSampler(
            model=args.served_model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            openai_api_base=args.url
        ),
    }

    if args.list_models:
        print("Available models:")
        for model_name in models.keys():
            print(f" - {model_name}")
        return []

    if args.model:
        if args.model not in models:
            print(f"Error: Model '{args.model}' not found.")
            return []
        models = {args.model: models[args.model]}

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        if eval_name == "mmlu":
            return MMLUEval(num_examples=1 if debug_mode else num_examples, num_threads=args.num_threads,
                            dataset_path=args.dataset_path)
        elif eval_name == "gpqa":
            return GPQAEval(
                n_repeats=1 if debug_mode else 10, num_examples=num_examples, num_threads=args.num_threads,
                dataset_path=args.dataset_path
            )
        elif eval_name == "mgsm":
            return MGSMEval(num_examples_per_lang=10 if debug_mode else 250, num_threads=args.num_threads,
                            dataset_path=args.dataset_path)
        elif eval_name == "drop":
            return DropEval(
                num_examples=10 if debug_mode else num_examples,
                train_samples_per_prompt=3, num_threads=args.num_threads, dataset_path=args.dataset_path
            )
        elif eval_name == "humaneval":
            return HumanEval(num_examples=10 if debug_mode else num_examples)
        else:
            raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {
        eval_name: get_evals(eval_name, args.debug)
        for eval_name in args.dataset
    }
    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    print(f"====================服务url: {args.url}")
    mergekey2resultpath = {}
    try:
        for model_name, sampler in models.items():
            for eval_name, eval_obj in evals.items():
                start_time = time.time()
                result = eval_obj(sampler)
                # ^^^ how to use a sampler
                now = datetime.datetime.now(tz=datetime.timezone.utc)
                formatted_time = now.strftime("%Y%m%d_%H%M%S")
                file_stem = f"{eval_name}_{model_name}_{formatted_time}"
                report_filename = f"{project_path}/results/{file_stem}{debug_suffix}.html"
                print(f"Writing report to {report_filename}")
                with open(report_filename, "w", encoding="utf-8") as fh:
                    fh.write(common.make_report(result))
                metrics = result.metrics | {"score": result.score}
                print(metrics)
                result_filename = f"{project_path}/results/{file_stem}{debug_suffix}.json"
                with open(result_filename, "w") as f:
                    f.write(json.dumps(metrics, indent=2))
                print(f"Writing results to {result_filename}")
                end_time = time.time()
                print(f"用例集: {eval_name}, 耗时：{end_time-start_time}")
                mergekey2resultpath[f"{eval_name}_{model_name}"] = result_filename
        merge_metrics = []
        for eval_model_name, result_filename in mergekey2resultpath.items():
            try:
                result = json.load(open(result_filename, "r+"))
            except Exception as e:
                print(e, result_filename)
                continue
            result = result.get("f1_score", result.get("score", None))
            eval_name = eval_model_name[: eval_model_name.find("_")]
            model_name = eval_model_name[eval_model_name.find("_") + 1:]
            merge_metrics.append(
                {"eval_name": eval_name, "model_name": model_name, "metric": result}
            )
        merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
            index=["model_name"], columns="eval_name"
        )
        print("\nAll results: ")
        print(merge_metrics_df.to_markdown())
        return merge_metrics
    except Exception as e:
        print(f"程序抛出未知异常：{str(e)}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
