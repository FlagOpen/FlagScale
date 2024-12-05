from omegaconf import OmegaConf
import argparse
import ray


task_config = OmegaConf.create()

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
    def decorator(func):
        # task_config.serve.deploy[0].models
        remote_func = ray.remote(*args, **kwargs)(func)

        def wrapper(*args, **kwargs):
            future = remote_func.remote(*args, **kwargs)
            result = ray.get(future)
            return result

        return wrapper

    return decorator


def load(config: OmegaConf) -> None:
    parser = argparse.ArgumentParser(description="Start vllm serve with Ray")

    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the model")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    global task_config
    task_config = OmegaConf.merge(task_config, config)
    return
