from omegaconf import OmegaConf
import ray


global_config = OmegaConf.create()

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
        remote_func = ray.remote(*args, **kwargs)(func)

        def wrapper(*args, **kwargs):            
            future = remote_func.remote(*args, **kwargs)
            result = ray.get(future)
            return result

        return wrapper

    return decorator


def load(config: OmegaConf) -> None:
    global global_config
    global_config = OmegaConf.merge(global_config, config)
    return
