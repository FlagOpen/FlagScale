import ray


def auto_remote(num_gpus=0):
    def decorator(cls):
        original_cls = ray.remote(num_gpus=num_gpus)(cls)

        class Wrapper:
            def __init__(self, *args, **kwargs):
                # Bypass __getattr__ by directly setting _actor in __dict__.
                object.__setattr__(self, "_actor", original_cls.remote(*args, **kwargs))

            def __getattr__(self, name):
                # Now we fetch the _actor *without* going through __getattr__ again.
                actor = object.__getattribute__(self, "_actor")

                def method(*args, **kwargs):
                    remote_method = getattr(actor, name)
                    return ray.get(remote_method.remote(*args, **kwargs))

                # Here we call getattr(...) on the *real* remote actor,
                # not on self (the Wrapper), so no infinite recursion.
                remote_method = getattr(actor, name)
                if name in ("generate", "gpu_computation"):
                    return method
                else:
                    return remote_method
        return Wrapper

    return decorator
