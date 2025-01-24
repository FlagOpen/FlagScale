import os
import torch
from llmcompressor.modifiers.utils.compression_wrapper import ModuleCompressionWrapper
from compressed_tensors.utils import update_parameter_data
from compressed_tensors.quantization.lifecycle import fake_quantize
from flagscale.compress.observers.base import Observer

__all__ = ["RTNWrapper"]
class RTNWrapper(ModuleCompressionWrapper):
    def __init__(self, name, layer, enable_fake_quant=False):
        super(RTNWrapper, self).__init__(name, layer)
        quantization_scheme = getattr(self.layer, "quantization_scheme", None)
        self._enable_fake_quant = enable_fake_quant
        self.weight_observer = None
        self.input_observer = None
        self.output_observer = None
        self.weights_observer_args = getattr(quantization_scheme, "weights", None)
        self.input_observer_args = getattr(quantization_scheme, "input_activations", None)
        self.output_observer_args = getattr(quantization_scheme, "output_activations", None)
        if self._enable_fake_quant:
            if self.input_observer_args and self.input_observer_args.dynamic:
                self.input_observer_args.observer = "minmax"
                self.input_observer = Observer.load_from_registry(self.input_observer_args.get_observer(), quantization_args=self.input_observer_args)
        else:
            if self.weights_observer_args and not self.weights_observer_args.dynamic:
                self.weight_observer = Observer.load_from_registry(self.weights_observer_args.get_observer(), quantization_args=self.weights_observer_args)
            if self.input_observer_args and not self.input_observer_args.dynamic:
                self.input_observer = Observer.load_from_registry(self.input_observer_args.get_observer(), quantization_args=self.input_observer_args)
            if self.output_observer_args and not self.output_observer_args.dynamic:
                self.output_observer = Observer.load_from_registry(self.output_observer_args.get_observer(), quantization_args=self.output_observer_args)

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        if self.input_observer:
            updated_scale, updated_zero_point = self.input_observer(inp)
            update_parameter_data(self.layer, updated_scale, f"input_scale")
            update_parameter_data(self.layer, updated_zero_point, f"input_zero_point")

    def compress(self, g_idx=None):
        if self.weight_observer:
            updated_scale, updated_zero_point = self.weight_observer(self.layer.weight, g_idx=g_idx)
            update_parameter_data(self.layer, updated_scale, f"weight_scale")
            update_parameter_data(self.layer, updated_zero_point, f"weight_zero_point")

    def enable_fake_quant(self):
        self._enable_fake_quant = True

    def forward(self, inp, **kwargs):
        """
        Run a forward pass of the wrapped layer
        """
        if self._enable_fake_quant:
            if self.input_observer_args:
                print("self.input_observer_args: ", self.input_observer_args)
                if self.input_observer_args.dynamic:
                    scale, zp = self.input_observer(inp)
                    tmp_inp = fake_quantize(inp, scale, zp, self.input_observer_args)
                    error = torch.nn.functional.mse_loss(inp, tmp_inp)
                    # print("input dynamic error: ", error, inp, tmp_inp, scale, zp)
                    inp = tmp_inp
                    del tmp_inp, error
                else:
                    inp = fake_quantize(inp, self.layer.input_scale, self.layer.input_zero_point, self.input_observer_args)
            if self.weights_observer_args:
                W = fake_quantize(self.layer.weight, self.layer.weight_scale, self.layer.weight_zero_point, self.weights_observer_args)
                update_parameter_data(self.layer, W, f"weight")
                del W
        out = self.layer(inp, **kwargs)
        # if self._enable_fake_quant and self.output_observer:
        #     out = fake_quantize(out, self.layer.output_scale, self.layer.output_zero_point, self.output_observer.quantization_args)
        torch.cuda.empty_cache()
        return out
