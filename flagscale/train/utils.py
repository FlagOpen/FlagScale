import sys
import importlib

_modules_to_modify = {"transformer_engine"}


class Empty:
    def __init__(self, *args, **kwargs):
        pass


def modify_module(module_name, module):
    if module_name == "transformer_engine":
        module.common.recipe = Empty()
        module.common.recipe.DelayedScaling = Empty()


class CustomModuleFinder:

    def find_module(self, module_name, path=None):
        if module_name in _modules_to_modify:
            return CustomModuleLoader()


class CustomModuleLoader:

    def load_module(self, module_name):
        if module_name in sys.modules:
            return sys.modules[module_name]

        original_finder = sys.meta_path.pop(0)
        module = importlib.import_module(module_name)
        modify_module(module_name, module)
        sys.meta_path.insert(0, original_finder)
        return module
