import operator
import torch
from torch.nn import Module
from llmcompressor.modifiers.utils.layer_compressor import LayerCompressor
from llmcompressor.utils.fsdp.context import (
    summon_full_params_context,
)
from llmcompressor.utils.pytorch.module import set_layer, get_layer

def replace_block(target: str, model: Module, target_module: Module):
    parent_target = ".".join(target.split(".")[:-1])
    parent_layer = get_layer(parent_target, model)[1]
    setattr(parent_layer, target.split(".")[-1], target_module)

class BlockCompressor(LayerCompressor):
    def pre_compress(self):
        # full_name = self._get_full_submodule_name(self.name)
        full_name = self.name
        # import pdb; pdb.set_trace()
        with summon_full_params_context(self.layer):
            wrapper = self.module_compressor_class(full_name, self.layer)
            if len(full_name) == 0:  # special case if layer has no children (i.e. lm_head)
                with summon_full_params_context(self.model):
                    replace_block(full_name, self.model, wrapper)
            else:
                replace_block(full_name, self.model, wrapper)
            self.modules[full_name] = wrapper

        self.layer = operator.attrgetter(self.name)(self.model)

        def add_batch(name):
            def tmp(_, inp, out):
                self.modules[name].add_batch(inp[0].data, out[0].data)

            return tmp

        for name in self.modules:
            self.handles.append(self.modules[name].register_forward_hook(add_batch(name)))

    def revert_layer_wrappers(self):
        """
        Reverts wrapped root modules back to their original structure
        """
        for name, module_wrapper in self.modules.items():
            full_name = self.name
            if len(full_name) == 0:  # special case if layer has no children (i.e. lm_head)
                with summon_full_params_context(self.model):
                    replace_block(full_name, self.model, module_wrapper.layer)
            else:
                replace_block(full_name, self.model, module_wrapper.layer)
            torch.cuda.empty_cache()
        self.modules = None