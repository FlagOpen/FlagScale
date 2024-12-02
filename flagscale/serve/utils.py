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
