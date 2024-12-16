from omegaconf import OmegaConf
import argparse
import ray


class TaskConfig:
    _instance = OmegaConf.create()
    _loaded = False

    @classmethod
    def load(cls, config=None):
        """
        Load the global configuration.
        This can only be called once. 
        
        :param config: An OmegaConf config object. If None, raise a RuntimeErrord.
        """
        if cls._loaded:
            return
        if config is None:
            raise RuntimeError("config must not be None.")

        cls._instance.update(config)
        cls._loaded = True

    @classmethod
    def get(cls):
        """
        Retrieve the loaded configuration.
        """
        return cls._instance


class TaskManager:
    def __init__(self):
        pass


def init(*args, **kwargs):
    _load()
    ray.init(*args, **kwargs)


def run():
    ray.run()


def stop():
    ray.shutdown()


def remote(*args, **kwargs):
    """Transform a function into a Ray task"""
    _load()
    def _merge_kwargs(func_name, **kwargs):
        new_kwargs = kwargs.copy()
        models = task_config.serve.deploy.models

        if func_name in models:
            new_kwargs.update(models[func_name])
            if "model_name" not in kwargs:
                new_kwargs.pop("model_name", None)

        return new_kwargs

    new_kwargs = _merge_kwargs(kwargs["name"], **kwargs)

    return ray.remote(*args, **new_kwargs)


def _load() -> None:
    """Load configuration for cluster init"""
    parser = argparse.ArgumentParser(description="Start vllm serve with Ray")

    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument("--log-dir", type=str, default="outputs", help="Path to the model")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    config.update({"log_dir": args.log_dir})
    TaskConfig.load(config)

    return


def prepare() -> None:
    # Load config
    # _load()
    return
