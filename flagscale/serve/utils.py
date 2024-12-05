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

    def _merge_kwargs(func_name, *args, **kwargs):
        new_kwargs = kwargs.copy()
        for instance in task_config.serve.deploy:
            for model in instance.models:
                if model.model_name == func_name:
                    new_kwargs.update(model)
                    new_kwargs.pop("model_name")
        return new_kwargs


    def decorator(func):
        new_kwargs = _merge_kwargs(func.__name__, *args, **kwargs)
        remote_func = ray.remote(*args, **new_kwargs)(func)

        def wrapper(*args, **kwargs):
            future = remote_func.remote(*args, **kwargs)
            result = ray.get(future)
            return result

        return wrapper

    return decorator


def _load(config: OmegaConf) -> None:
    """Load configuration for cluster init"""
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

def prepare() -> None:
    # Load config
    _load()
    pass