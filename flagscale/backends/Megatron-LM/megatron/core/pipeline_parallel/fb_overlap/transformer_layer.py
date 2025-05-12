#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from contextlib import nullcontext
import torch
from .modules.utils import (
    NoopLayerGraph, LayerGraph, is_p2p_comm_needed,
    p2p_comm_helper, P2PCommOutput, P2PCommParams
)

from .overlap_funcs import (
    transformer_layer_forward_moe,
    transformer_layer_forward_dense,
    transformer_layer_backward_moe,
    transformer_layer_backward_dense,
    transformer_layer_backward_noop,
    transformer_layer_forward_moe_backward_moe_overlaping,
    transformer_layer_forward_dense_backward_moe_overlaping,
    transformer_layer_forward_moe_backward_dense_overlaping,
    transformer_layer_forward_dense_backward_dense_overlaping,
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
    print(f"in transformer_layer_forward, start")
    if checkpoint:
        checkpoint_context = torch.no_grad()
    else:
        checkpoint_context = nullcontext()

    with checkpoint_context:
        layer_forward_func = None
        if use_orig_layer_forward:
            from mindspeed.core.pipeline_parallel.fp_overlap.megatron_adaptor import get_orig_transformer_layer_forward_func
            # for mtp transformer layer forward
            layer_forward_func = get_orig_transformer_layer_forward_func()
            return layer_forward_func(
                self, hidden_states, attention_mask,
                context, context_mask, rotary_pos_emb, inference_params, packed_seq_params
            )
        elif hasattr(self.mlp, 'experts'):
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
    print(f"in transformer_layer_backward, start")
    if layer_graph.checkpointed:
        with torch.enable_grad():
            _, _, restored_layer_graph = transformer_layer_forward(
                layer_graph.layer, layer_graph.layer_input, *layer_graph.layer_inputs, checkpoint=False
            )
            restored_layer_graph.unperm2_graph = (restored_layer_graph.unperm2_graph[0], layer_graph.unperm2_graph[1])
            layer_graph = restored_layer_graph
    if isinstance(layer_graph, NoopLayerGraph):
        return transformer_layer_backward_noop(layer_output_grad, layer_graph)
    elif layer_graph.is_moe_layer:
        return transformer_layer_backward_moe(layer_output_grad, layer_graph)
    else:
        return transformer_layer_backward_dense(layer_output_grad, layer_graph)



def transformer_layer_forward_backward_overlaping(
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
    print(f"in transformer_layer_forward_backward_overlaping, start")
    if bwd_layer_graph is None or isinstance(bwd_layer_graph, NoopLayerGraph):
        # no f&w overlaping
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
            print(f"in transformer_layer_forward_backward_overlaping, return 1")
            return out
        else:
            output, context, graph = transformer_layer_forward(
                fwd_layer, hidden_states, attention_mask, context, context_mask, rotary_pos_emb,
                inference_params, packed_seq_params, use_orig_layer_forward, checkpoint=checkpoint
            )
            # handle fwd p2p communication
            next_iter_input_tensor, fwd_p2p_handles = None, None
            fwd_pp_comm_params = pp_comm_params
            if is_p2p_comm_needed(fwd_pp_comm_params):
                next_iter_input_tensor, fwd_p2p_handles = p2p_comm_helper(fwd_pp_comm_params, output)

            bwd_input_grad = transformer_layer_backward(bwd_layer_output_grad, bwd_layer_graph)
            next_iter_output_tensor_grad, bwd_p2p_handles = None, None
            if bwd_input_grad is not None:
                # handle bwd p2p communication
                if is_p2p_comm_needed(bwd_pp_comm_params):
                    next_iter_output_tensor_grad, bwd_p2p_handles = p2p_comm_helper(bwd_pp_comm_params, bwd_input_grad)

            if checkpoint:
                graph.record_layer_inputs(
                    attention_mask, context, context_mask, rotary_pos_emb,
                    inference_params, packed_seq_params, use_orig_layer_forward
                )
            print(f"in transformer_layer_forward_backward_overlaping, return 2")
            return (output, context, graph,
                    (bwd_input_grad, None),
                    P2PCommOutput(next_iter_input_tensor, next_iter_output_tensor_grad, fwd_p2p_handles, bwd_p2p_handles, bwd_input_grad))

    else:
        fb_overlap_func = None
        if hasattr(fwd_layer.mlp, 'experts') and bwd_layer_graph.is_moe_layer:
            fb_overlap_func = transformer_layer_forward_moe_backward_moe_overlaping
        elif hasattr(fwd_layer.mlp, 'experts') and not bwd_layer_graph.is_moe_layer:
            fb_overlap_func = transformer_layer_forward_moe_backward_dense_overlaping
        elif not hasattr(fwd_layer.mlp, 'experts') and bwd_layer_graph.is_moe_layer:
            fb_overlap_func = transformer_layer_forward_dense_backward_moe_overlaping
        elif not hasattr(fwd_layer.mlp, 'experts') and not bwd_layer_graph.is_moe_layer:
            fb_overlap_func = transformer_layer_forward_dense_backward_dense_overlaping
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
        print(f"in transformer_layer_forward_backward_overlaping, return 3")
        return out
