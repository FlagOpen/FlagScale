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

    def _merge_kwargs(func_name, **kwargs):
        new_kwargs = kwargs.copy()
        models = task_config.serve.deploy.models

        if func_name in models:
            new_kwargs.update(models[func_name])
            if "model_name" not in kwargs:
                new_kwargs.pop("model_name", None)

        return new_kwargs
    
    assert len(args) == 1 and len(kwargs) == 0 and callable(args[0]), f"Invalid arguments with args: {args} and kargs {kwargs}"

    new_kwargs = _merge_kwargs(kwargs[], **kwargs)

    return ray.remote(*args, **new_kwargs)


def _load() -> None:
    """Load configuration for cluster init"""
    parser = argparse.ArgumentParser(description="Start vllm serve with Ray")

    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the model")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    global task_config
    task_config.update(config)
    task_config.update({"log_dir": args.log_dir})

    return


def prepare() -> None:
    # Load config
    _load()
    return
