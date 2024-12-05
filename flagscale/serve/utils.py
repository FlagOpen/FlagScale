import time
import logging
import ray


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
    def decorator(func):
        remote_func = ray.remote(*args, **kwargs)(func)

        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            future = remote_func.remote(*args, **kwargs)
            
            result = ray.get(future)
            
            end_time = time.time()
            
            return result

        return wrapper

    return decorator
