# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Some codes are adapted from Huawei MindSpore

from contextlib import nullcontext
import torch
from .modules.utils import (
    LayerGraph, P2PCommParams
)

from .overlap_funcs.fwd import (
    transformer_layer_forward_moe,
    transformer_layer_forward_dense,
)

from .overlap_funcs.bwd import (
    transformer_layer_backward_moe,
    transformer_layer_backward_dense,
)

from .overlap_funcs.fwdbwd import (
    transformer_layer_forward_moe_backward_moe_overlapping,
    transformer_layer_forward_dense_backward_moe_overlapping,
    transformer_layer_forward_moe_backward_dense_overlapping,
    transformer_layer_forward_dense_backward_dense_overlapping,
)


def transformer_layer_forward(
    self,
    hidden_states,
    attention_mask,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    inference_params=None,
    packed_seq_params=None,
    use_orig_layer_forward=False,
    checkpoint=False
):
    """ Forward function of transformer layer, dense or moe
    """
    if checkpoint:
        checkpoint_context = torch.no_grad()
    else:
        checkpoint_context = nullcontext()

    with checkpoint_context:
        layer_forward_func = None
        if hasattr(self.mlp, 'experts'):
            layer_forward_func = transformer_layer_forward_moe
        else:
            layer_forward_func = transformer_layer_forward_dense

        return layer_forward_func(
            self, hidden_states, attention_mask,
            context, context_mask, rotary_pos_emb, inference_params, packed_seq_params, checkpoint=checkpoint
        )


def transformer_layer_backward(
    layer_output_grad,
    layer_graph
):
    """ Backward function of transformer layer, dense or moe
    """
    if layer_graph.checkpointed:
        with torch.enable_grad():
            _, _, restored_layer_graph = transformer_layer_forward(
                layer_graph.layer, layer_graph.layer_input, *layer_graph.layer_inputs, checkpoint=False
            )
            restored_layer_graph.unperm2_graph = (restored_layer_graph.unperm2_graph[0], layer_graph.unperm2_graph[1])
            layer_graph = restored_layer_graph
    
    if layer_graph.is_moe_layer:
        return transformer_layer_backward_moe(layer_output_grad, layer_graph)
    else:
        return transformer_layer_backward_dense(layer_output_grad, layer_graph)


def transformer_layer_forward_backward_overlapping(
    fwd_layer,
    hidden_states,
    attention_mask,
    bwd_layer_output_grad=None,
    bwd_layer_graph: LayerGraph = None,
    bwd_unperm_a2a_handle=None,
    next_bwd_layer_graph: LayerGraph = None,
    context=None,
    context_mask=None,
    rotary_pos_emb=None,
    inference_params=None,
    packed_seq_params=None,
    pp_comm_params: P2PCommParams = None,
    bwd_pp_comm_params: P2PCommParams = None,
    use_orig_layer_forward=False,
    checkpoint=False
):
    """ Forward-backward overlapping function of transformer layer, dense or moe
    if bwd_layer_graph is None, execute forward without overlapping;
    if bwd_layer_graph is not None, there are four possible combinations:
    [forward dense, backward dense], [forward dense, backward moe], [forward moe, backward dense], [forward moe, backward moe]
    """
    if bwd_layer_graph is None:
        out = transformer_layer_forward(
            fwd_layer, hidden_states, attention_mask, context, context_mask, rotary_pos_emb,
            inference_params, packed_seq_params, use_orig_layer_forward, checkpoint=checkpoint
        )
        if len(out) > 2 and checkpoint:
            out[2].record_layer_inputs(
                attention_mask, context, context_mask, rotary_pos_emb,
                inference_params, packed_seq_params, use_orig_layer_forward
            )
        return out

    else:
        fb_overlap_func = None
        if hasattr(fwd_layer.mlp, 'experts') and bwd_layer_graph.is_moe_layer:
            fb_overlap_func = transformer_layer_forward_moe_backward_moe_overlapping
        elif hasattr(fwd_layer.mlp, 'experts') and not bwd_layer_graph.is_moe_layer:
            fb_overlap_func = transformer_layer_forward_moe_backward_dense_overlapping
        elif not hasattr(fwd_layer.mlp, 'experts') and bwd_layer_graph.is_moe_layer:
            fb_overlap_func = transformer_layer_forward_dense_backward_moe_overlapping
        elif not hasattr(fwd_layer.mlp, 'experts') and not bwd_layer_graph.is_moe_layer:
            fb_overlap_func = transformer_layer_forward_dense_backward_dense_overlapping
        else:
            raise AssertionError('Check Layer Spec, f&b overlap func is not supported!')

        if bwd_layer_graph.checkpointed:
            _, _, bwd_layer_graph = transformer_layer_forward(
                bwd_layer_graph.layer, bwd_layer_graph.layer_input, *bwd_layer_graph.layer_inputs, checkpoint=False
            )

        out = fb_overlap_func(
            fwd_layer, hidden_states, attention_mask, bwd_layer_output_grad, bwd_layer_graph, bwd_unperm_a2a_handle,
            next_bwd_layer_graph, context, context_mask, rotary_pos_emb, inference_params,
            packed_seq_params, pp_comm_params, bwd_pp_comm_params, checkpoint=checkpoint
        )

        if checkpoint:
            out[2].record_layer_inputs(
                attention_mask, context, context_mask, rotary_pos_emb,
                inference_params, packed_seq_params, use_orig_layer_forward
            )
        return out

