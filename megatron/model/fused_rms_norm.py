# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import importlib

from megatron.core.utils import make_viewless_tensor

from apex.normalization.fused_layer_norm import FusedRMSNormAffineFunction


class MixedFusedRMSNorm(torch.nn.Module):

  def __init__(self, normalized_shape, eps=1e-5,
               no_persist_layer_norm=True,
               sequence_parallel=False,
               apply_layernorm_1p=False,
               init_weight=None):
        super(MixedFusedRMSNorm, self).__init__()

        self.apply_layernorm_1p = apply_layernorm_1p
        assert not (self.apply_layernorm_1p and self.apply_layernorm_rms), \
            "Cannot apply both 1p and rms layernorm"

        self.init_weight = init_weight
        assert self.init_weight is None or isinstance(self.init_weight, float), \
            "Cannot init_weight of None or of non-float"
        assert not (self.init_weight is not None and self.apply_layernorm_1p), \
            "Cannot float init_weight and 1p layernorm"


        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [1024, 1536, 2048, 2304, 3072, 3840, 4096,
            5120, 6144, 8192, 10240, 12288, 12800, 15360, 16384, 18432, 20480,
            24576, 25600, 30720, 32768, 40960, 49152, 65536]
        if normalized_shape not in persist_ln_hidden_sizes or \
                not HAVE_PERSIST_LAYER_NORM:
            no_persist_layer_norm = True

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()
        self.no_persist_layer_norm = no_persist_layer_norm
        self.sequence_parallel = sequence_parallel

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)


  def reset_parameters(self):

    if self.apply_layernorm_1p:
        init.zeros_(self.weight)
        init.zeros_(self.bias)
    else:
        if self.init_weight:
            init.constant_(self.weight, self.init_weight)
        else:
            init.ones_(self.weight)
        if not self.apply_layernorm_rms:
            init.zeros_(self.bias)

  def forward(self, input):

    weight = self.weight + 1 if self.apply_layernorm_1p else self.weight

    return FusedRMSNormAffineFunction.apply(input, weight, self.normalized_shape, self.eps)