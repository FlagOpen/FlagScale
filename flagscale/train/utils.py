import sys
import importlib

_hook_modules = {"transformer_engine"}

def module_hook(fullname, module):
    if fullname == "transformer_engine":
        print("module_hook fullname:", fullname)
        module.common.recipe.DelayedScaling = None


class MetaPathFinder:

    def find_module(self, fullname, path=None):
        print('find_module {}'.format(fullname))
        if fullname in _hook_modules:
            return MetaPathLoader()


class MetaPathLoader:

    def load_module(self, fullname):
        print('load_module {}'.format(fullname))
        if fullname in sys.modules:
            return sys.modules[fullname]

        finder = sys.meta_path.pop(0)
        module = importlib.import_module(fullname)
        module_hook(fullname, module)
        sys.meta_path.insert(0, finder)
        return module
