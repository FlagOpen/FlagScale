# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Custom normalization layers."""
import torch
import torch_npu
from omni.models.common.config.model_config import model_extra_config
from typing import Optional, Union, Any
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNormGPU


class RMSNorm(RMSNormGPU):
    def forward(
            self,
            x: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
            quant_symbol: bool = False,
    ) -> Union[tuple[dict[str, Any], Any], Any]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            if model_extra_config.operator_opt_config.use_w8a8_dynamic_quant and quant_symbol:
                x_int8, pertoken_scale = torch_npu.npu_dynamic_quant(x)
                x = {"x_int8": x_int8, "pertoken_scale": pertoken_scale}
            return x, residual

        return torch_npu.npu_rms_norm(
            x,
            self.weight.data,
            self.variance_epsilon,
        )[0]