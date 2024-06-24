import sys
import importlib
import os
_hook_modules = {"transformer_engine"}

device_type = os.environ.get('DEVICE_TYPE',None)

class Empty:
    def __init__(self,*args):    
       pass

def module_hook(fullname, module):
    if device_type == 'iluvatar':
        if fullname == "transformer_engine":
            print("module_hook fullname:", fullname)
            module.common.recipe = Empty()
            module.common.recipe.DelayedScaling = Empty()

class CustomModuleFinder:

    def find_module(self, fullname, path=None):
        if fullname in _hook_modules:
            return CustomModuleLoader()

class CustomModuleLoader:

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        finder = sys.meta_path.pop(0)
        module = importlib.import_module(fullname)
        module_hook(fullname, module)
        sys.meta_path.insert(0, finder)
        return module
