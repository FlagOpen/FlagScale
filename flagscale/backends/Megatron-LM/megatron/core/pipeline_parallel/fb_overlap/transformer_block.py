# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import List
from contextlib import nullcontext
from megatron.training import get_args
from megatron.core.utils import make_sharded_tensor_for_checkpoint, make_viewless_tensor
from .modules.utils import (
    detach_tensor, LayerGraph, P2PCommParams
)
from .transformer_layer import transformer_layer_forward, transformer_layer_backward, transformer_layer_forward_backward_overlaping


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
    # print(f"in transformer_block_forward, start")
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
            # hidden_states, context, saved_graphs = transformer_layer_forward(
            #     layer,
            #     hidden_states=hidden_states,
            #     attention_mask=attention_mask,
            #     context=context,
            #     context_mask=context_mask,
            #     rotary_pos_emb=rotary_pos_emb,
            #     inference_params=inference_params,
            #     packed_seq_params=packed_seq_params,
            #     checkpoint=checkpoint
            # )
            layer_graphs.append(saved_graphs)

    # Final layer norm.
    if self.post_process and self.post_layer_norm and self.final_layernorm is not None:
        detached_hidden_states = detach_tensor(hidden_states)
        layer_graphs[-1].unperm2_graph = (layer_graphs[-1].unperm2_graph[0], detached_hidden_states)
        hidden_states = self.final_layernorm(detached_hidden_states)

    # print(f"in transformer_block_forward, return")
    return (hidden_states, layer_graphs)


def transformer_block_backward(
    block_output_grad,
    layer_graphs: List[LayerGraph],
):
    # print(f"in transformer_block_backward, start")
    # should call backward fisrt for final_layernorm and postprocess grad
    layer_output_grad = block_output_grad
    while len(layer_graphs) > 0:
        layer_graph = layer_graphs.pop(-1)
        layer_output_grad = transformer_layer_backward(layer_output_grad, layer_graph)
    # print(f"in transformer_block_backward, return")
    return layer_output_grad


def transformer_block_forward_backward_overlaping(
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
    # print(f"in transformer_block_forward_backward_overlapping, start")
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

    rng_context = nullcontext()
    fp8_context = nullcontext()

    assert not fwd_block.config.enable_cuda_graph
    fwd_layer_graphs = []

    bwd_layer_output_grad = bwd_block_output_grad
    bwd_unperm_a2a_handle = None

    fwd_hidden_states, fwd_context = hidden_states, context
    with (((rng_context and fp8_context))):
        for l_no, fwd_layer in enumerate(fwd_block.layers):
            checkpoint = False
            if fwd_block.config.recompute_granularity == 'full' and fwd_block.training:
                if fwd_block.config.recompute_method == 'block':
                    recompute_skip_num_layers = 0
                    if fwd_block.config.fp8 and not hidden_states.requires_grad:
                        recompute_skip_num_layers += 1
                    if (l_no >= recompute_skip_num_layers and l_no < fwd_block.config.recompute_num_layers + recompute_skip_num_layers):
                        checkpoint = True
                if fwd_block.config.recompute_method == 'uniform':
                    assert fwd_block.config.recompute_num_layers == 1
                    checkpoint = True
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
                transformer_layer_forward_backward_overlaping(
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

    # Final layer norm.
    if fwd_block.post_process and fwd_block.post_layer_norm and fwd_block.final_layernorm is not None:
        detached_hidden_states = detach_tensor(fwd_hidden_states)
        fwd_layer_graphs[-1].unperm2_graph = (fwd_layer_graphs[-1].unperm2_graph[0], detached_hidden_states)
        fwd_hidden_states = fwd_block.final_layernorm(detached_hidden_states)

    # print(f"in transformer_block_forward_backward_overlaping, return")
    return (fwd_hidden_states, fwd_layer_graphs), bwd_layer_output_grad, pp_comm_output
