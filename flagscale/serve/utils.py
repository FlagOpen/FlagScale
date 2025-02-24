import argparse
import threading
from functools import wraps

import ray
from omegaconf import OmegaConf

task_config = OmegaConf.create()


def load_once(func):
    loaded = False
    lock = threading.Lock()

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal loaded
        with lock:
            if not loaded:
                func(*args, **kwargs)
                loaded = True

    return wrapper


@load_once
def load_args() -> None:
    """Load configuration for cluster init"""
    parser = argparse.ArgumentParser(description="Start vllm serve with Ray")

    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--log-dir", type=str, default="outputs", help="Path to the model"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    global task_config
    task_config.update(config)
    task_config.update({"log_dir": args.log_dir})

    return


class TaskManager:
    def __init__(self):
        pass


def init():
    ray.init(address="auto")


def run():
    ray.run()


def stop():
    ray.shutdown()


def remote(*args, **kwargs):
    """Transform a function into a Ray task"""
    load_args()

    def _merge_kwargs(func_name, **kwargs):
        new_kwargs = kwargs.copy()
        models = task_config.serve.deploy.get("models", None)
        if models and func_name in models:
            new_kwargs.update(models[func_name])
            if "model_name" not in kwargs:
                new_kwargs.pop("model_name", None)

        return new_kwargs

    new_kwargs = _merge_kwargs(kwargs["name"], **kwargs)

    return ray.remote(*args, **new_kwargs)
