# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Mainly adopted from https://gitee.com/ascend/MindSpeed/blob/master/mindspeed/core/transformer/moe/moe_feature/fb_overlap/transformer_block.py

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    get_transformer_layer_offset,
)
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor

try:
    import transformer_engine.pytorch as te  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex  # pylint: disable=unused-import

    HAVE_APEX = True
except ImportError:
    HAVE_APEX = False

get_cpu_offload_context = None
te_checkpoint = None

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import (
        TENorm,
        get_cpu_offload_context,
        te_checkpoint,
    )

    LayerNormImpl = TENorm

elif HAVE_APEX:
    LayerNormImpl = FusedLayerNorm

else:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    LayerNormImpl = WrappedTorchNorm

from megatron.training import get_args
from .modules.utils import (
    detach_tensor, LayerGraph, P2PCommParams
)
from .transformer_layer import transformer_layer_backward, transformer_layer_forward_backward_overlapping


def transformer_block_forward(
    self,
    hidden_states,
    attention_mask,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    inference_params=None,
    packed_seq_params=None,
):
    """ Forward function of transformer block
    """
    if not self.pre_process:
        # See set_input_tensor()
        hidden_states = self.input_tensor

    # Viewless tensor.
    # - We only need to create a viewless tensor in the case of micro batch
    #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
    #   above creates a view tensor, and '.contiguous()' is a pass-through.
    #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
    #   the need to make it viewless.
    #
    #   However, we don't explicitly check mbs == 1 here because
    #   make_viewless_tensor() has negligible overhead when its input
    #   is already viewless.
    #
    # - For the 'else' case above, calling make_viewless_tensor() here is
    #   likely redundant, since p2p_communication.py (likely originator)
    #   already creates viewless tensors. That said, make_viewless_tensor()
    #   is called here to be future-proof and corner-case-proof.
    hidden_states = make_viewless_tensor(
        inp=hidden_states,
        requires_grad=True,
        keep_graph=True,
    )

    rng_context = nullcontext()
    fp8_context = nullcontext()

    assert not self.config.enable_cuda_graph
    layer_graphs = []
    args = get_args()

    with rng_context and fp8_context:
        for l_no, layer in enumerate(self.layers):
            checkpoint = False
            if self.config.recompute_granularity == 'full' and self.training:
                if self.config.recompute_method == 'block':
                    recompute_skip_num_layers = 0
                    if self.config.fp8 and not hidden_states.requires_grad:
                        recompute_skip_num_layers += 1
                    if (l_no >= recompute_skip_num_layers and l_no < self.config.recompute_num_layers + recompute_skip_num_layers):
                        checkpoint = True
                if self.config.recompute_method == 'uniform':
                    assert self.config.recompute_num_layers == 1
                    checkpoint = True
            with self.offload_context:
                hidden_states, context, saved_graphs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    inference_params=inference_params,
                    packed_seq_params=packed_seq_params,
                    checkpoint=checkpoint
                )
                layer_graphs.append(saved_graphs)
            if (
                torch.is_grad_enabled()
                and self.config.cpu_offloading
                and self.group_prefetch_offload_commit_async is not None
            ):
                hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

    # Final layer norm.
    if self.post_process and self.post_layer_norm and self.final_layernorm is not None:
        detached_hidden_states = detach_tensor(hidden_states)
        layer_graphs[-1].unperm2_graph = (layer_graphs[-1].unperm2_graph[0], detached_hidden_states)
        hidden_states = self.final_layernorm(detached_hidden_states)

    return (hidden_states, layer_graphs)


def transformer_block_backward(
    block_output_grad,
    layer_graphs: List[LayerGraph],
):
    """ Backward function of transformer block
    """
    layer_output_grad = block_output_grad
    while len(layer_graphs) > 0:
        layer_graph = layer_graphs.pop(-1)
        layer_output_grad = transformer_layer_backward(layer_output_grad, layer_graph)
    return layer_output_grad


def transformer_block_forward_backward_overlapping(
    fwd_block,
    hidden_states,
    attention_mask,
    bwd_block_output_grad,
    bwd_block_graphs: List[LayerGraph],
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    inference_params=None,
    packed_seq_params=None,
    pp_comm_params: P2PCommParams = None,
    bwd_pp_comm_params: P2PCommParams = None,
):
    """ Forward-backward overlapping function of transformer block
    """
    # Delete the obsolete reference to the initial input tensor if necessary
    if isinstance(hidden_states, WrappedTensor):
        hidden_states = hidden_states.unwrap()

    if not fwd_block.pre_process:
        # See set_input_tensor()
        hidden_states = fwd_block.input_tensor

    # Viewless tensor.
    # - We only need to create a viewless tensor in the case of micro batch
    #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
    #   above creates a view tensor, and '.contiguous()' is a pass-through.
    #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
    #   the need to make it viewless.
    #
    #   However, we don't explicitly check mbs == 1 here because
    #   make_viewless_tensor() has negligible overhead when its input
    #   is already viewless.
    #
    # - For the 'else' case above, calling make_viewless_tensor() here is
    #   likely redundant, since p2p_communication.py (likely originator)
    #   already creates viewless tensors. That said, make_viewless_tensor()
    #   is called here to be future-proof and corner-case-proof.
    hidden_states = make_viewless_tensor(
        inp=hidden_states,
        requires_grad=True,
        keep_graph=True,
    )

    if fwd_block.config.sequence_parallel:
        rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
    else:
        rng_context = nullcontext()

    # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
    # otherwise do nothing extra at the outer level
    # if we are using other fp8 recipes, then the context manager enter&exit are free
    # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
    # control which layer will be fp8 or bf16
    use_outer_fp8_context = fwd_block.config.fp8 and fwd_block.config.fp8_recipe == Fp8Recipe.delayed
    use_inner_fp8_context = fwd_block.config.fp8 and fwd_block.config.fp8_recipe != Fp8Recipe.delayed
    outer_fp8_context = get_fp8_context(fwd_block.config) if use_outer_fp8_context else nullcontext()

    assert not fwd_block.config.enable_cuda_graph
    fwd_layer_graphs = []

    bwd_layer_output_grad = bwd_block_output_grad
    bwd_unperm_a2a_handle = None

    fwd_hidden_states, fwd_context = hidden_states, context
    with rng_context, outer_fp8_context:
        for l_no, fwd_layer in enumerate(fwd_block.layers):
            inner_fp8_context = (
                get_fp8_context(fwd_block.config, layer.layer_number - 1)
                if use_inner_fp8_context
                else nullcontext()
            )
            with fwd_block.offload_context, inner_fp8_context:
                checkpoint = False
                bwd_layer_graph = bwd_block_graphs.pop(-1)
                cur_p2p_params = pp_comm_params
                cur_bwd_p2p_params = bwd_pp_comm_params
                if l_no != len(fwd_block.layers) - 1 or len(bwd_block_graphs) > 0:
                    # no need to excute pp communication in the intermediate layers
                    cur_p2p_params = P2PCommParams()
                    cur_bwd_p2p_params = P2PCommParams()
                next_bwd_layer_graph = None
                if (len(bwd_block_graphs) > 0 and
                    not bwd_block_graphs[-1].checkpointed and
                    l_no != len(fwd_block.layers) - 1
                ):
                    next_bwd_layer_graph = bwd_block_graphs[-1]
            
                fwd_hidden_states, fwd_context, fwd_layer_graph, \
                (bwd_layer_output_grad, bwd_unperm_a2a_handle), \
                pp_comm_output = \
                    transformer_layer_forward_backward_overlapping(
                        fwd_layer,
                        fwd_hidden_states,
                        attention_mask,
                        bwd_layer_output_grad,
                        bwd_layer_graph=bwd_layer_graph,
                        bwd_unperm_a2a_handle=bwd_unperm_a2a_handle,
                        next_bwd_layer_graph=next_bwd_layer_graph,
                        context=fwd_context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_params=inference_params,
                        packed_seq_params=packed_seq_params,
                        pp_comm_params=cur_p2p_params,
                        bwd_pp_comm_params=cur_bwd_p2p_params,
                        checkpoint=checkpoint
                    )
                fwd_layer_graphs.append(fwd_layer_graph)
            
            if (
                torch.is_grad_enabled()
                and fwd_block.config.cpu_offloading
                and fwd_block.group_prefetch_offload_commit_async is not None
            ):
                fwd_hidden_states = fwd_block.group_prefetch_offload_commit_async(fwd_hidden_states)

    # Final layer norm.
    if fwd_block.post_process and fwd_block.post_layer_norm and fwd_block.final_layernorm is not None:
        detached_hidden_states = detach_tensor(fwd_hidden_states)
        fwd_layer_graphs[-1].unperm2_graph = (fwd_layer_graphs[-1].unperm2_graph[0], detached_hidden_states)
        fwd_hidden_states = fwd_block.final_layernorm(detached_hidden_states)

    return (fwd_hidden_states, fwd_layer_graphs), bwd_layer_output_grad, pp_comm_output
