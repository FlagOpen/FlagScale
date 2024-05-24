# this file is used for adding tools func  to processing patches 

def add_patches_module(path: str, module_dict: dict):
    if len(module_dict) == 0:
        raise Exception(f"module dict is None")
    import sys
    print(f"{path} is being instead, using module {module_dict}")
    for k in sys.modules:
        if k.startswith(path):
            for module_name  in module_dict.keys():
                if getattr(sys.modules[k], module_name, None):
                    setattr(sys.modules[k], module_name, module_dict[module_name])

def add_patches_func(path: str, func_dict: dict):
    if len(func_dict) == 0:
        raise Exception(f"module dict is None")
    import sys
    print(f"{path} is being instead, using function {func_dict}")
    for k in sys.modules:
        if k.startswith(path):
            for func_name, func in func_dict.items():
                if getattr(sys.modules[k], func_name, None):
                    setattr(sys.modules[k], func_name, func)        





