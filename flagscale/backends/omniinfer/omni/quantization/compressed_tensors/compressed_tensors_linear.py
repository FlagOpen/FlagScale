# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Callable, Any, Dict, List, Optional, Union

import torch
from torch.nn import Parameter
import torch_npu

from compressed_tensors.quantization import QuantizationStrategy
from vllm.distributed import tensor_model_parallel_all_gather
from omni.models.common.config.model_config import model_extra_config

BEFORE_INIT = 0
AFTER_INIT = 1
WEIGHT_BITS = 8
ACTIVATION_BITS = 8

ASCEND_ALIGN_BYTES = 512
INT_8_BYTES = 1

class AscendCompressedTensorsW8A8Int8LinearMethod:
    """Linear method for Ascend W8A8_DYNAMIC.
    """

    def __init__(self):
        self.transpose_weight = True

    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype, output_size: int, strategy: str) -> Dict[str, Any]:
        if strategy != QuantizationStrategy.TENSOR:
            return {}
        weight_scale = torch.empty(
            len(output_size), dtype=torch.float32 if params_dtype == torch.float16 else torch.bfloat16)
        weight_offset = None
        params_dict = {}
        params_dict["weight_scale"] = weight_scale
        params_dict["weight_offset"] = weight_offset
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
        strategy: str,
    ) -> Dict[str, Any]:
        if strategy != QuantizationStrategy.CHANNEL:
            return {}
        weight_scale = torch.empty((output_size, 1),
                             dtype=torch.float32 if params_dtype == torch.float16 else torch.bfloat16)
        weight_offset = torch.zeros((output_size, 1),
                             dtype=torch.float32 if params_dtype == torch.float16 else torch.bfloat16)
        params_dict = {}
        params_dict["weight_scale"] = weight_scale
        params_dict["weight_offset"] = weight_offset
        return params_dict

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        inner_gather: bool = False
    ) -> torch.Tensor:
        if layer.init_state == BEFORE_INIT:
            layer.init_state = AFTER_INIT

        # activation per-token dynamic quant
        if isinstance(x, Dict):
            x_int8 = x.get('x_int8')
            pertoken_scale = x.get('pertoken_scale')
        else:
            x_int8, pertoken_scale = torch_npu.npu_dynamic_quant(x)

        if model_extra_config.operator_opt_config.enable_pd_separated and inner_gather:
            x_int8 = tensor_model_parallel_all_gather(x_int8, dim=0)
            pertoken_scale = tensor_model_parallel_all_gather(pertoken_scale, dim=0)

        throw_dequant = getattr(layer, 'throw_dequant', False)
        if throw_dequant:
            assert bias is None
            return (torch_npu.npu_quant_matmul(x_int8, layer.weight, layer.weight_scale,
                                            bias=None, output_dtype=torch.int32),
                    pertoken_scale)
        out = torch_npu.npu_quant_matmul(x_int8, layer.weight, layer.weight_scale,
                                offset=None,
                                pertoken_scale=pertoken_scale,
                                bias=bias,
                                output_dtype=torch.bfloat16)
        return out

    def process_weights_after_loading(self, layer):
        weight = layer.weight
        weight_scale = layer.weight_scale
        
        if getattr(layer, 'throw_dequant', False):
            weight_scale = weight_scale.to(torch.float32)
        weight_offset = layer.weight_offset
        weight = torch_npu.npu_format_cast(weight.t().contiguous(), 29)
        layer.weight = Parameter(weight, requires_grad=False)

        layer.weight_scale = Parameter(weight_scale.view(-1), requires_grad=False)
        layer.weight_offset = Parameter(weight_offset.view(-1).float(), requires_grad=False)
