#!/usr/bin/env python
# coding=utf-8
import megatron

try:
    from torch_musa import _ext as fused_weight_gradient_mlp_cuda
    _grad_accum_fusion_available = True
except ImportError:
    _grad_accum_fusion_available = False

megatron.core.tensor_parallel.layers.fused_weight_gradient_mlp_cuda = fused_weight_gradient_mlp_cuda 
megatron.core.tensor_parallel.layers._grad_accum_fusion_available = True
