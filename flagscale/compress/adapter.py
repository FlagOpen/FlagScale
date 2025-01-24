import re
import os

import torch
from torch.nn import Module
from llmcompressor.modifiers.quantization.gptq.utils.gptq_wrapper import GPTQWrapper
from llmcompressor.modifiers.utils.layer_compressor import LayerCompressor
from llmcompressor.utils.fsdp.context import fix_fsdp_module_name
from llmcompressor.modifiers.utils.pytorch_helpers import run_calibration_forward
from compressed_tensors.quantization import (
    QuantizationScheme, 
    QuantizationStatus, 
    QuantizationConfig, 
    is_preset_scheme, 
    preset_name_to_scheme, 
    apply_quantization_config
    )
from compressed_tensors.quantization.lifecycle.apply import find_name_or_class_matches
from llmcompressor.modifiers.quantization.gptq.utils import get_output_error
from llmcompressor.utils.helpers import DisableKVCache
from compressed_tensors.quantization import (
    QuantizationScheme,
    disable_quantization,
    enable_quantization,
    is_attention_module,
)
from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    calibrate_input_hook,
    calibrate_kv_cache_input_hook,
    calibrate_kv_cache_output_hook,
    calibrate_output_hook,
    freeze_module_quantization,
    initialize_observer,
    set_unset_kv_cache,
    update_weight_zp_scale,
)
from llmcompressor.transformers.sparsification.compressed_tensors_utils import modify_save_pretrained
from flagscale.compress.algo import SmoothQuantWrapper, RTNWrapper
from flagscale.compress.blockwise_compressor import BlockCompressor

from flagscale.runner.runner_utils import logger
import pdb

__all__ = ["LLMCompressorAdapter"]

QUANT_MAPPING_NAMES = {
        "gptq": GPTQWrapper,
    }

BLOCKWISE_WRAPPER_NAMES = {
    "smoothquant": SmoothQuantWrapper,
}

class LLMCompressorAdapter:
    def __init__(self, model, scheme=None, targets=None, algo=None, ignore=None, dataset=None, num_calibration_steps=384):
        self.model = model
        # print("model: ", model)
        # modify_save_pretrained(self.model)
        if algo is not None:
            assert len(algo) == 1
            for k, v in algo.items():
                self.algo = k
                self.algo_args = v
        else:
            self.algo = algo
            self.algo_args = {}
        self.scheme = scheme
        self.ignore = ignore
        self.targets = targets
        self.num_calibration_steps = num_calibration_steps 
        self.dataset = dataset
        self.config_groups = None
        self.wrapper_cls = None
        self.compress_granularity = None
        self.layer_compressors_ = []
        self.require_calib = True

        support_algos = list(QUANT_MAPPING_NAMES.keys()) + list(BLOCKWISE_WRAPPER_NAMES.keys())
        if (self.algo is None and is_preset_scheme(self.scheme)) or self.algo in support_algos:
            if self.algo is not None:
                if self.algo in QUANT_MAPPING_NAMES:
                    self.wrapper_cls = QUANT_MAPPING_NAMES[self.algo]
                    self.compress_granularity = LayerCompressor  
                elif self.algo in BLOCKWISE_WRAPPER_NAMES:
                    self.wrapper_cls = BLOCKWISE_WRAPPER_NAMES[self.algo]
                    self.compress_granularity = BlockCompressor
                else:
                    raise f"algorithm: {self.algo} not implemented"
            else:
                self.wrapper_cls = RTNWrapper
                self.compress_granularity = LayerCompressor
        quant_config = self.init_quant_config()
        print(quant_config)

        if quant_config is not None:
            ### find ignore and target to quant, initialize module for quant
            ### overwrite forward if quantization_enabled is Tue
            apply_quantization_config(self.model, quant_config)
            self.require_calib = quant_config.requires_calibration_data()

        self.init_compressor()
        if self.require_calib:
            # self.insert_observer()
            if model.training == False: ### Post Training
                assert self.dataset is not None, f"The algorithm {self.algo} you selected requires a calibration process, please provide the calibration data"
                self.run_blockwise_calib_forward()
                self.model.apply(freeze_module_quantization)
            else:  ### Training Aware
                pass
        else:
            self.layer_compressors_[0].clear_early_stop()
            for idx, layer_compressor in enumerate(self.layer_compressors_):
                layer_compressor.pre_compress()
                # import pdb;pdb.set_trace()
                layer_compressor.compress()
                layer_compressor.post_compress()
                layer_compressor.revert_layer_wrappers()
        
    def init_quant_config(self):
        if self.scheme is not None:
            # takes precedence over config_groups
            if isinstance(self.scheme, str) and is_preset_scheme(self.scheme):
                # attach targets to scheme
                self.scheme = {self.scheme: self.targets}

            self.config_groups = {}
            for idx, key in enumerate(self.scheme.keys()):
                if is_preset_scheme(key):
                    scheme = preset_name_to_scheme(key, self.scheme[key])
                else:
                    scheme = QuantizationScheme.model_validate(
                        {"targets": self.scheme[key], **self.scheme}
                    )

                group_name = f"group_{idx}"
                self.config_groups[group_name] = scheme

            if self.config_groups is None or len(self.config_groups) == 0:
                default_quant_scheme = QuantizationScheme(targets=self.targets)
                self.config_groups = {"group_0": default_quant_scheme}
                logger.info(
                    f"No config groups were provided, using default {self.config_groups}"
                )

            return QuantizationConfig(
                config_groups=self.config_groups,
                kv_cache_scheme=None, ### TODO(lvmengsi): not support kv cache quant for now
                quantization_status=QuantizationStatus.INITIALIZED,
                ignore=self.ignore,
            )
        else:
            return None

    def init_compressor(self):
        for name, layer in self.model.named_modules():
            name = fix_fsdp_module_name(name)
            if name is None:
                continue
            try:
                idx = int(name.split(".")[-1])
            except:
                continue

            if matches := find_name_or_class_matches(name, layer, self.ignore):
                continue
            logger.info(f"prepare compressor for layer {name}")
            compressor = self.compress_granularity(self.wrapper_cls, self.model, layer, idx, name, self.algo_args)
            self.layer_compressors_.append(compressor)
        self.layer_compressors_[0].set_early_stop()

    @torch.no_grad()
    def run_blockwise_calib_forward(self):
        logger.info(f"start calibration")
        self.model.apply(disable_quantization)
        with DisableKVCache(self.model):
            intermediates = run_calibration_forward(
                    self.model, self.dataset, num_calibration_steps=self.num_calibration_steps, mask_padding=False
                )
            self.layer_compressors_[0].clear_early_stop()
            
            for idx, layer_compressor in enumerate(self.layer_compressors_):
                logger.info(f"start calibration layer {layer_compressor.name}")
                layer_compressor.pre_compress()
                # print("idx: ", idx, intermediates)
                unquantized_outputs = layer_compressor.calibrate_layer(intermediates)
                layer_compressor.compress()
                layer_compressor.post_compress()
                layer_compressor.revert_layer_wrappers()
                quantized_outputs = layer_compressor.calibrate_layer(intermediates)
                error = get_output_error(unquantized_outputs, quantized_outputs)
                logger.info(f"Mean output error from quantization: {error:.3f}")
                intermediates = quantized_outputs
        self.model.apply(enable_quantization)
    
