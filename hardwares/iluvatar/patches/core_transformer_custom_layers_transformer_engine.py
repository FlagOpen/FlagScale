import dataclasses
import os
from importlib.metadata import version
from typing import Callable

import torch
import transformer_engine as te
from pkg_resources import packaging
from torch import Tensor

from megatron.core import ModelParallelConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_global_ranks,
    get_context_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from flagscale.patches_utils import add_patches_module

#[iluvatar] start of changes
def forward(self, x):
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        out = super().forward(x, is_first_microbatch=_is_first_microbatch)
        self.is_first_microbatch = False

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        return out


class TEDelayedScaling:
  def __init__(
        self,
        config: ModelParallelConfig,
        fp8_format: int,
        override_linear_precision: tuple = (False, False, False),
    ):
        extra_kwargs = _get_extra_te_kwargs(config)
        if _te_version >= packaging.version.Version("1.6.0.dev0"):
            extra_kwargs["fp8_dpa"] = config.fp8_dot_product_attention
            extra_kwargs["fp8_mha"] = config.fp8_multi_head_attention
        try:
            super().__init__(
            margin=config.fp8_margin,
            interval=config.fp8_interval,
            fp8_format=fp8_format,
            amax_compute_algo=config.fp8_amax_compute_algo,
            amax_history_len=config.fp8_amax_history_len,
            override_linear_precision=override_linear_precision,
            **extra_kwargs,
            )
        except:
           print("TEDelayedScaling initialize failed, seems like your gpu doesn't support fp8 configuration, ignore TEDelayedScaling here")
#[iluvatar] end of changes

module_path = "megatron.core.transformer.custom_layers.transformer_engine"
module_dict = {"TELinear.forward":forward,
               "TELayerNormColumnParallelLinear.forward":forward,
               "TEDelayedScaling":TEDelayedScaling}
add_patches_module(module_path,module_dict)
