#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import importlib
import sys
import types
from typing import Any, Dict, List, Optional

from vllm.logger import logger

from .compressed_tensors.compressed_tensors_linear import AscendCompressedTensorsW8A8Int8LinearMethod
from .compressed_tensors.compressed_tensors_moe import AscendCompressedTensorsW8A8Int8MoEMethod

CUSTOMIZED_QUANTIZER_TYPE: List[str] = []


class AscendQuantizer:
    _instance: Optional[object] = None

    def __init__(self, quant_description):
        pass

    @staticmethod
    def get_linear_method():
        raise NotImplementedError(
            "Linear method is not implemented for the current quant type.")

    @staticmethod
    def get_moe_method():
        raise NotImplementedError(
            "MoE method is not implemented for the current quant type.")

    @staticmethod
    def get_attention_method():
        raise NotImplementedError(
            "Attention method is not implemented for the current quant type.")

    @staticmethod
    def get_linear_quant_type(quant_description: Dict[str, Any], prefix: str,
                              packed_modules_mapping: Dict[str, Any]):
        proj_name = prefix.split(".")[-1]
        if proj_name in packed_modules_mapping:
            quant_type = None
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in packed_modules_mapping[proj_name]
            ]
            for shard_prefix in shard_prefixes:
                shard_quant_type = quant_description[shard_prefix + '.weight']

                if quant_type is None:
                    quant_type = shard_quant_type
                elif shard_quant_type != quant_type:
                    raise ValueError(
                        f"Not all shards of {prefix} are quantized with same quant type."
                        f"Shard {proj_name} uses {shard_quant_type}, but another shard"
                        f"use {quant_type}. Please check quantization config.")
        else:
            quant_type = quant_description[prefix + '.weight']
        return quant_type

    @classmethod
    def get_quantizer(cls,
                      quant_description: Dict[str, Any],
                      prefix: str,
                      packed_modules_mapping: Optional[Dict[str, Any]] = None):
        quant_type = quant_description.get("quant_method", "")
        if quant_type not in SUPPORT_ASCEND_QUANTIZER_MATHOD:
            if packed_modules_mapping is None:
                packed_modules_mapping = dict()
            # Attention
            if '.attn' in prefix and 'fa_quant_type' in quant_description.keys():
                quant_type = quant_description['fa_quant_type']
            # Linear
            else:
                quant_type = cls.get_linear_quant_type(quant_description, prefix,
                                                    packed_modules_mapping)
        if quant_type in SUPPORT_ASCEND_QUANTIZERS.keys():
            cls = SUPPORT_ASCEND_QUANTIZERS[quant_type]
            if not cls._instance:
                cls._instance = cls(quant_description)
            return cls._instance
        raise NotImplementedError("Currently, vLLM Ascend only supports following quant types:" \
                                  f"{list(SUPPORT_ASCEND_QUANTIZERS.keys())}")



class CompressedTensorsQuantizer(AscendQuantizer):
    @staticmethod
    def get_linear_method():
        return AscendCompressedTensorsW8A8Int8LinearMethod()
 
    @staticmethod
    def get_moe_method():
        return AscendCompressedTensorsW8A8Int8MoEMethod()

SUPPORT_ASCEND_QUANTIZER_MATHOD = [
    "compressed-tensors",
]

SUPPORT_ASCEND_QUANTIZERS = {
    "compressed-tensors": CompressedTensorsQuantizer,
}
