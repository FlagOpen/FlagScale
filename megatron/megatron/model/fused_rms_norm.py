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

        self.init_weight = init_weight
        assert self.init_weight is None or isinstance(self.init_weight, float), \
            "Cannot init_weight of None or of non-float"


        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()
        self.sequence_parallel = sequence_parallel

        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)


  def reset_parameters(self):
    if self.init_weight:
        init.constant_(self.weight, self.init_weight)
    else:
        init.ones_(self.weight)

  def forward(self, input):

    weight = self.weight

    return FusedRMSNormAffineFunction.apply(input, weight, self.normalized_shape, self.eps)