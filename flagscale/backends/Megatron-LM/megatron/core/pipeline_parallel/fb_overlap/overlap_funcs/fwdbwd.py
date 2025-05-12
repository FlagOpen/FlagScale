#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from contextlib import nullcontext
import torch
from megatron.core.utils import make_sharded_tensor_for_checkpoint, make_viewless_tensor
from megatron.core import parallel_state, tensor_parallel
from megatron.training import get_args
from megatron.core.transformer.moe.moe_utils import permute
from megatron.core.pipeline_parallel.fb_overlap.modules.utils import async_all_gather, async_all_to_all, async_reduce_scatter, AG_SHARED_EXPERTS_INPUTS
from ..modules.token_dispatcher import (
    alltoall_token_perm1, alltoall_token_perm2,
    alltoall_token_unperm1, alltoall_token_unperm2
)
from ..modules.weight_grad_store import WeightGradStore
from ..modules.attention import (
    attention_forward,
)
from ..modules.utils import (
    detach_tensor, run_graph_backward, LayerGraph, is_p2p_comm_needed,
    p2p_comm_helper, P2PCommOutput, P2PCommParams
)


def router_forward(
    self,
    hidden_states
):
    probs, indices = self.mlp.router(hidden_states)

    return probs, indices


def transformer_layer_forward_dense_backward_moe_overlaping(
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
    checkpoint=False
):
    print(f"in transformer_layer_forward_dense_backward_moe_overlaping, start")
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if checkpoint:
        checkpoint_context = torch.no_grad()
    else:
        checkpoint_context = nullcontext()
    args = get_args()
    ep_group = parallel_state.get_expert_model_parallel_group()
    recomp_norm = getattr(args, 'recompute_norm', False)
    bwd_dispached_input, bwd_fc1_out, bwd_act_out, bwd_probs, bwd_indices, global_input_tokens_local_experts_indices = bwd_layer_graph.recompute_needed_tensors

    # Unperm2 Bwd
    # check if backward unpermutation alltoall is launched at bwd layer before
    if bwd_unperm_a2a_handle is None:
        run_graph_backward(bwd_layer_graph.unperm2_graph, bwd_layer_output_grad)
        # Async Unperm A2A
        _, unperm1_out_grad, bwd_unperm_a2a_handle = async_all_to_all(
            bwd_layer_graph.unperm_a2a_graph[1].grad,
            bwd_layer_graph.output_splits,
            bwd_layer_graph.input_splits,
            ep_group
        )
    else:
        unperm1_out_grad = bwd_layer_output_grad

    # # skip recompute
    # if args.moe_zero_memory == 'level0':
    #     ...

    with checkpoint_context:
        # Atten Fwd
        detached_layer_input = detach_tensor(hidden_states, checkpoint_forward=checkpoint)

        # Residual connection.
        residual1 = detached_layer_input

        # input_layernorm + AttentionForward
        hidden_states = attention_forward(
            fwd_layer, detached_layer_input, residual1,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            recompute_norm=recomp_norm
        )

        attention_out, detached_attention_out = hidden_states, detach_tensor(hidden_states,
                                                                               checkpoint_forward=checkpoint)

        # Residual connection.
        residual2 = detached_attention_out

        if recomp_norm:
            assert not recomp_norm, "not support recompute norm"
            pre_mlp_layernorm_output = fwd_layer.norm_ckpt2.checkpoint(fwd_layer.pre_mlp_layernorm, False,
                                                                       detached_attention_out)
        else:
            pre_mlp_layernorm_output = fwd_layer.pre_mlp_layernorm(detached_attention_out)

    # # skip recompute
    # if args.moe_zero_memory == 'level0':
    #     ...

    # # skip recompute
    # if args.moe_zero_memory == 'level0' or should_recompute_activation(bwd_layer_graph.layer.layer_number):
    #     ...


    bwd_unperm_a2a_handle.wait()
    bwd_unperm_a2a_handle = None
    run_graph_backward(bwd_layer_graph.unperm1_graph, unperm1_out_grad)
    unperm1_out_grad.untyped_storage().resize_(0)
    WeightGradStore.start_decouple()
    run_graph_backward(bwd_layer_graph.grouped_mlp_graph, keep_grad=True)  # keep for dw
    WeightGradStore.end_decouple()
    run_graph_backward(bwd_layer_graph.perm2_graph, keep_graph=True)  # keep for dw
    run_graph_backward(bwd_layer_graph.perm2_append_graph, keep_graph=True)
    # # skip recompute
    # if args.moe_zero_memory == 'level0':
    #     ...

    if tp_size > 1:
        shared_expert_grad = bwd_layer_graph.shared_experts_graph[1].grad
        _, backward_ag_shared, backward_ag_shared_handle = async_all_gather(
            shared_expert_grad, parallel_state.get_tensor_model_parallel_group()
        )
    else:
        backward_ag_shared = bwd_layer_graph.shared_experts_graph[1].grad
        backward_ag_shared_handle = None

    _, perm1_out1_grad, bwd_perm_a2a_handle1 = async_all_to_all(
        bwd_layer_graph.perm_a2a_graph[1].grad,
        bwd_layer_graph.input_splits,
        bwd_layer_graph.output_splits,
        ep_group,
    )
    _, perm1_out2_grad, bwd_perm_a2a_handle2 = async_all_to_all(
        bwd_layer_graph.perm_a2a_append_graph[1].grad,
        bwd_layer_graph.input_splits,
        bwd_layer_graph.output_splits,
        ep_group,
    )

    # Grouped MLP dw computation

    with checkpoint_context:
        # MLP Forward
        detached_mlp_input = detach_tensor(pre_mlp_layernorm_output, checkpoint_forward=checkpoint)
        mlp_output_with_bias = fwd_layer.mlp(detached_mlp_input)
        if recomp_norm:
            assert not recomp_norm, "not support recompute norm"
            fwd_layer.norm_ckpt2.discard_output()
            mlp_output_with_bias[0].register_hook(fwd_layer.norm_ckpt2.recompute)

    bwd_perm_a2a_handle1.wait()
    bwd_perm_a2a_handle1 = None
    bwd_perm_a2a_handle2.wait()
    bwd_perm_a2a_handle2 = None
    run_graph_backward(bwd_layer_graph.perm1_graph, perm1_out1_grad)
    run_graph_backward(bwd_layer_graph.perm1_append_graph, perm1_out2_grad)
    perm1_out1_grad.untyped_storage().resize_(0)
    perm1_out2_grad.untyped_storage().resize_(0)
    WeightGradStore.start_decouple()
    if backward_ag_shared_handle is not None:
        backward_ag_shared_handle.wait()
        backward_ag_shared_handle = None
        shared_expert_grad.untyped_storage().resize_(0)
    run_graph_backward(bwd_layer_graph.shared_experts_graph, backward_ag_shared, keep_grad=True)  # dw computation
    WeightGradStore.end_decouple()
    run_graph_backward(bwd_layer_graph.router_graph)
    run_graph_backward(bwd_layer_graph.pre_mlp_layernorm_graph, keep_graph=True)
    WeightGradStore.start_decouple()
    run_graph_backward(bwd_layer_graph.attn_graph, keep_grad=True)
    WeightGradStore.end_decouple()

    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        run_graph_backward(next_bwd_layer_graph.unperm2_graph, bwd_layer_graph.layer_input.grad, keep_graph=True)

    next_layer_output_grad, next_bwd_unperm_a2a_handle = bwd_layer_graph.layer_input.grad, None
    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        _, next_layer_output_grad, next_bwd_unperm_a2a_handle = async_all_to_all(
            next_bwd_layer_graph.unperm_a2a_graph[1].grad,
            next_bwd_layer_graph.output_splits,
            next_bwd_layer_graph.input_splits,
            ep_group
        )

    with checkpoint_context:
        with fwd_layer.bias_dropout_add_exec_handler():
            hidden_states = fwd_layer.mlp_bda(fwd_layer.training, fwd_layer.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual2, fwd_layer.hidden_dropout
            )

    output = make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )

    # handle fwd p2p communication
    next_iter_input_tensor, fwd_p2p_handles = None, None
    fwd_pp_comm_params = pp_comm_params
    if is_p2p_comm_needed(fwd_pp_comm_params):
        next_iter_input_tensor, fwd_p2p_handles = p2p_comm_helper(fwd_pp_comm_params, output)

    # handle bwd p2p communication
    next_iter_output_tensor_grad, bwd_p2p_handles = None, None
    if is_p2p_comm_needed(bwd_pp_comm_params):
        next_iter_output_tensor_grad, bwd_p2p_handles = p2p_comm_helper(bwd_pp_comm_params, bwd_layer_graph.layer_input.grad)

    # # skip recompute
    # if args.moe_zero_memory == 'level0':
    #     ...

    WeightGradStore.pop()

    saved_tensors = (
        (attention_out, detached_attention_out),
        (pre_mlp_layernorm_output, detached_mlp_input),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),  # perm2 graph
        (None, None),
        (None, None),  # grouped mlp graph
        (None, None),  # unperm1 graph
        (None, None),
        (output, None),  # unperm2 graph
        (None, None),
        detached_layer_input
    )

    graph = LayerGraph(
        saved_tensors, [], None, None, fwd_layer,
        checkpointed=checkpoint
    )

    for tensor in bwd_layer_graph.recompute_needed_tensors:
        if tensor is not None:
            tensor.untyped_storage().resize_(0)
    print(f"in transformer_layer_forward_dense_backward_moe_overlaping, return")
    return (output, context, graph,
            (next_layer_output_grad, next_bwd_unperm_a2a_handle),
            P2PCommOutput(next_iter_input_tensor, next_iter_output_tensor_grad, fwd_p2p_handles, bwd_p2p_handles, bwd_layer_graph.layer_input.grad))



def transformer_layer_forward_moe_backward_dense_overlaping(
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
    checkpoint=False
):
    print(f"in transformer_layer_forward_moe_backward_dense_overlaping, start")
    args = get_args()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    use_shared_experts = hasattr(fwd_layer.mlp, 'shared_experts') and fwd_layer.mlp.shared_experts is not None
    if checkpoint:
        checkpoint_context = torch.no_grad()
    else:
        checkpoint_context = nullcontext()
    args = get_args()
    ep_group = parallel_state.get_expert_model_parallel_group()
    recomp_norm = getattr(args, 'recompute_norm', False)

    with checkpoint_context:
        # Atten Fwd
        detached_layer_input = detach_tensor(hidden_states, checkpoint_forward=checkpoint)

        # Residual connection.
        residual1 = detached_layer_input

        # input_layernorm + AttentionForward
        hidden_states = attention_forward(
            fwd_layer, detached_layer_input, residual1,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            recompute_norm=recomp_norm
        )

        attention_out, detached_attention_out = hidden_states, detach_tensor(hidden_states)

        # Residual connection.
        residual2 = detached_attention_out

        if recomp_norm:
            assert not recomp_norm, "not support recompute norm"
            pre_mlp_layernorm_output = fwd_layer.norm_ckpt2.checkpoint(fwd_layer.pre_mlp_layernorm, False, detached_attention_out)
        else:
            pre_mlp_layernorm_output = fwd_layer.pre_mlp_layernorm(detached_attention_out)

        # MLP.
        detached_mlp_input = detach_tensor(pre_mlp_layernorm_output, checkpoint_forward=checkpoint)
        probs, routing_map = router_forward(fwd_layer, detached_mlp_input)

        # Token Permutation Forward
        probs_detached = detach_tensor(probs, checkpoint_forward=checkpoint)
        perm1_local_input_tokens, perm1_probs, tokens_per_expert = alltoall_token_perm1(fwd_layer.mlp.token_dispatcher, detached_mlp_input, probs_detached, routing_map)

        _, perm1_local_input_tokens_a2a, perm1_local_input_tokens_a2a_handle = async_all_to_all(
            perm1_local_input_tokens,
            fwd_layer.mlp.token_dispatcher.output_splits,
            fwd_layer.mlp.token_dispatcher.input_splits,
            ep_group
        )

        _, perm1_probs_a2a, perm1_probs_a2a_handle = async_all_to_all(
            perm1_probs,
            fwd_layer.mlp.token_dispatcher.output_splits,
            fwd_layer.mlp.token_dispatcher.input_splits,
            ep_group
        )

    WeightGradStore.start_decouple()
    run_graph_backward(bwd_layer_graph.unperm2_graph, bwd_layer_output_grad, keep_grad=True)  # keep for dw
    run_graph_backward(bwd_layer_graph.pre_mlp_layernorm_graph, keep_graph=True)
    WeightGradStore.end_decouple()

    perm1_local_input_tokens_a2a_handle.wait()
    perm1_local_input_tokens_a2a_handle = None
    perm1_probs_a2a_handle.wait()
    perm1_probs_a2a_handle = None
    perm1_local_input_tokens.untyped_storage().resize_(0)
    perm1_probs.untyped_storage().resize_(0)

    # Grouped MLP dw computation

    with checkpoint_context:
        detached_perm1_local_input_tokens_a2a = detach_tensor(perm1_local_input_tokens_a2a, checkpoint_forward=checkpoint)
        detached_perm1_probs_a2a = detach_tensor(perm1_probs_a2a, checkpoint_forward=checkpoint)
        dispatched_input, tokens_per_expert, permuted_probs = alltoall_token_perm2(fwd_layer.mlp.token_dispatcher, detached_perm1_local_input_tokens_a2a, detached_perm1_probs_a2a, tokens_per_expert)
        perm1_local_input_tokens_a2a.untyped_storage().resize_(0)
        perm1_probs_a2a.untyped_storage().resize_(0)

        if tp_size > 1 and use_shared_experts:
            _, shared_experts_input, shared_experts_allgather_handle = async_all_gather(
                detached_mlp_input, tp_group, is_use_get_global_memory_buffer=True
            )
            AG_SHARED_EXPERTS_INPUTS.append((shared_experts_input, shared_experts_allgather_handle))
        else:
            shared_experts_input, shared_experts_allgather_handle = detached_mlp_input, None

        # Grouped MLP Forward
        detached_dispatched_input = detach_tensor(dispatched_input, checkpoint_forward=checkpoint)
        detached_permuted_probs = detach_tensor(permuted_probs, checkpoint_forward=checkpoint)
        expert_output, mlp_bias = fwd_layer.mlp.experts(detached_dispatched_input, tokens_per_expert, detached_permuted_probs)
        # # skip recompute
        # if args.moe_zero_memory == 'level0':
        #     ...
        
        recompute_needed_tensors = [None, None, None, None, None, None]
        detached_expert_output = detach_tensor(expert_output, checkpoint_forward=checkpoint)
        detached_mlp_bias = detach_tensor(mlp_bias, checkpoint_forward=checkpoint)
        
        # Token Unpermutaion Forward
        unperm1_hidden_states = alltoall_token_unperm1(fwd_layer.mlp.token_dispatcher, detached_expert_output, detached_mlp_bias)
        expert_output.untyped_storage().resize_(0)
        if shared_experts_allgather_handle is not None:
            shared_experts_allgather_handle.wait()
            shared_experts_allgather_handle = None
        
        # Launch Token Unperm2 A2A
        _, unperm1_hidden_states_a2a, unperm1_hidden_states_a2a_handle = async_all_to_all(
            unperm1_hidden_states,
            fwd_layer.mlp.token_dispatcher.input_splits,
            fwd_layer.mlp.token_dispatcher.output_splits,
            ep_group
        )

        share_experts_graph = None
        if use_shared_experts:
            shared_expert_output = fwd_layer.mlp.shared_experts(detached_mlp_input)
            if tp_size > 1:
                share_experts_graph, shared_expert_output, rs_shared_experts_handle = async_reduce_scatter(
                    shared_expert_output, tp_group
                )
                rs_shared_experts_handle.wait()
                rs_shared_experts_handle = None
                share_experts_graph.untyped_storage().resize_(0)
            else:
                share_experts_graph = shared_expert_output

        if recomp_norm:
            assert not recomp_norm, "not support recompute norm"
            fwd_layer.norm_ckpt2.discard_output()

    WeightGradStore.start_decouple()
    run_graph_backward(bwd_layer_graph.attn_graph, keep_grad=True)
    WeightGradStore.end_decouple()

    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        run_graph_backward(next_bwd_layer_graph.unperm2_graph, bwd_layer_graph.layer_input.grad, keep_graph=True)

    unperm1_hidden_states_a2a_handle.wait()
    unperm1_hidden_states_a2a_handle = None
    unperm1_hidden_states.untyped_storage().resize_(0)

    next_layer_output_grad, next_bwd_unperm_a2a_handle = bwd_layer_graph.layer_input.grad, None
    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        _, next_layer_output_grad, next_bwd_unperm_a2a_handle = async_all_to_all(
            next_bwd_layer_graph.unperm_a2a_graph[1].grad,
            next_bwd_layer_graph.output_splits,
            next_bwd_layer_graph.input_splits,
            ep_group
        )

    with checkpoint_context:
        detached_unperm1_hidden_states_a2a = detach_tensor(unperm1_hidden_states_a2a, checkpoint_forward=checkpoint)
        route_expert_output, _ = alltoall_token_unperm2(fwd_layer.mlp.token_dispatcher, detached_unperm1_hidden_states_a2a)

        if hasattr(fwd_layer.mlp, 'shared_experts') and fwd_layer.mlp.shared_experts is not None:
            detached_shared_expert_output = detach_tensor(shared_expert_output, checkpoint_forward=checkpoint)
            mlp_output = route_expert_output + detached_shared_expert_output
            shared_expert_output.untyped_storage().resize_(0)
        else:
            detached_shared_expert_output = None
            mlp_output = route_expert_output

        if recomp_norm:
            assert not recomp_norm, "not support recompute norm"
            mlp_output.register_hook(fwd_layer.norm_ckpt2.recompute)


        with fwd_layer.bias_dropout_add_exec_handler():
            hidden_states = fwd_layer.mlp_bda(fwd_layer.training, fwd_layer.config.bias_dropout_fusion)(
                (mlp_output, None), residual2, fwd_layer.hidden_dropout
            )

    output = make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )

    # handle fwd p2p communication
    next_iter_input_tensor, fwd_p2p_handles = None, None
    fwd_pp_comm_params = pp_comm_params
    if is_p2p_comm_needed(fwd_pp_comm_params):
        next_iter_input_tensor, fwd_p2p_handles = p2p_comm_helper(fwd_pp_comm_params, output)

    # handle bwd p2p communication
    next_iter_output_tensor_grad, bwd_p2p_handles = None, None
    if is_p2p_comm_needed(bwd_pp_comm_params):
        next_iter_output_tensor_grad, bwd_p2p_handles = p2p_comm_helper(bwd_pp_comm_params, bwd_layer_graph.layer_input.grad)

    WeightGradStore.pop()

    saved_tensors = (
        (attention_out, detached_attention_out),
        (pre_mlp_layernorm_output, detached_mlp_input),
        (probs, probs_detached),
        (perm1_local_input_tokens, None),  # perm1 graph
        (perm1_probs, None), # perm1 append graph
        (None, detached_perm1_local_input_tokens_a2a),
        (None, detached_perm1_probs_a2a),
        (dispatched_input, detached_dispatched_input),  # perm2 graph
        (permuted_probs, detached_permuted_probs), # perm2 append graph
        (expert_output, detached_expert_output),  # grouped mlp graph
        (unperm1_hidden_states, None),  # unperm1 graph
        (None, detached_unperm1_hidden_states_a2a),
        (output, None),  # unperm2 graph
        (share_experts_graph, detached_shared_expert_output),
        detached_layer_input
    )

    graph = LayerGraph(
        saved_tensors, recompute_needed_tensors,
        fwd_layer.mlp.token_dispatcher.input_splits, fwd_layer.mlp.token_dispatcher.output_splits, fwd_layer,
        checkpointed=checkpoint
    )

    for tensor in bwd_layer_graph.recompute_needed_tensors:
        if tensor is not None:
            tensor.untyped_storage().resize_(0)
    
    print(f"in transformer_layer_forward_moe_backward_dense_overlaping, return")
    return (output, context, graph,
            (next_layer_output_grad, next_bwd_unperm_a2a_handle),
            P2PCommOutput(next_iter_input_tensor, next_iter_output_tensor_grad, fwd_p2p_handles, bwd_p2p_handles, bwd_layer_graph.layer_input.grad))



def transformer_layer_forward_dense_backward_dense_overlaping(
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
    checkpoint=False
):
    print(f"in transformer_layer_forward_dense_backward_dense_overlaping, start")
    if checkpoint:
        checkpoint_context = torch.no_grad()
    else:
        checkpoint_context = nullcontext()
    args = get_args()
    ep_group = parallel_state.get_expert_model_parallel_group()
    recomp_norm = getattr(args, 'recompute_norm', False)

    with checkpoint_context:
        # Atten Fwd
        detached_layer_input = detach_tensor(hidden_states, checkpoint_forward=checkpoint)

        # Residual connection.
        residual1 = detached_layer_input

        # input_layernorm + AttentionForward
        hidden_states = attention_forward(
            fwd_layer, detached_layer_input, residual1,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            recompute_norm=recomp_norm
        )

        attention_out, detached_attention_out = hidden_states, detach_tensor(hidden_states, checkpoint_forward=checkpoint)

        # Residual connection.
        residual2 = detached_attention_out

        if recomp_norm:
            assert not recomp_norm, "not support recompute norm"
            pre_mlp_layernorm_output = fwd_layer.norm_ckpt2.checkpoint(fwd_layer.pre_mlp_layernorm, False, detached_attention_out)
        else:
            pre_mlp_layernorm_output = fwd_layer.pre_mlp_layernorm(detached_attention_out)

        # MLP.
        detached_mlp_input = detach_tensor(pre_mlp_layernorm_output, checkpoint_forward=checkpoint)
        mlp_output_with_bias = fwd_layer.mlp(detached_mlp_input)
        if recomp_norm:
            assert not recomp_norm, "not support recompute norm"
            fwd_layer.norm_ckpt2.discard_output()
            mlp_output_with_bias[0].register_hook(fwd_layer.norm_ckpt2.recompute)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with fwd_layer.bias_dropout_add_exec_handler():
            hidden_states = fwd_layer.mlp_bda(fwd_layer.training, fwd_layer.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual2, fwd_layer.hidden_dropout
            )

    output = make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )

    # handle fwd p2p communication
    next_iter_input_tensor, fwd_p2p_handles = None, None
    fwd_pp_comm_params = pp_comm_params
    if is_p2p_comm_needed(fwd_pp_comm_params):
        next_iter_input_tensor, fwd_p2p_handles = p2p_comm_helper(fwd_pp_comm_params, output)

    # Detach backward into dx/dw
    WeightGradStore.start_decouple()
    run_graph_backward(bwd_layer_graph.unperm2_graph, bwd_layer_output_grad, keep_grad=True)  # keep for dw
    run_graph_backward(bwd_layer_graph.pre_mlp_layernorm_graph, keep_graph=True)
    run_graph_backward(bwd_layer_graph.attn_graph, keep_grad=True)
    WeightGradStore.end_decouple()

    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        run_graph_backward(next_bwd_layer_graph.unperm2_graph, bwd_layer_graph.layer_input.grad, keep_graph=True)

    next_layer_output_grad, next_bwd_unperm_a2a_handle = bwd_layer_graph.layer_input.grad, None
    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        _, next_layer_output_grad, next_bwd_unperm_a2a_handle = async_all_to_all(
            next_bwd_layer_graph.unperm_a2a_graph[1].grad,
            next_bwd_layer_graph.output_splits,
            next_bwd_layer_graph.input_splits,
            ep_group
        )

    # handle bwd p2p communication
    next_iter_output_tensor_grad, bwd_p2p_handles = None, None
    if is_p2p_comm_needed(bwd_pp_comm_params):
        next_iter_output_tensor_grad, bwd_p2p_handles = p2p_comm_helper(bwd_pp_comm_params, bwd_layer_graph.layer_input.grad)

    WeightGradStore.pop()

    saved_tensors = (
        (attention_out, detached_attention_out),
        (pre_mlp_layernorm_output, detached_mlp_input),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),  # perm2 graph
        (None, None),
        (None, None),  # grouped mlp graph
        (None, None),  # unperm1 graph
        (None, None),
        (output, None),  # unperm2 graph
        (None, None),
        detached_layer_input
    )

    graph = LayerGraph(
        saved_tensors, [], None, None, fwd_layer,
        checkpointed=checkpoint
    )

    for tensor in bwd_layer_graph.recompute_needed_tensors:
        if tensor is not None:
            tensor.untyped_storage().resize_(0)

    print(f"in transformer_layer_forward_dense_backward_dense_overlaping, return")
    return (output, context, graph,
            (next_layer_output_grad, next_bwd_unperm_a2a_handle),
            P2PCommOutput(next_iter_input_tensor, next_iter_output_tensor_grad, fwd_p2p_handles, bwd_p2p_handles, bwd_layer_graph.layer_input.grad))


def transformer_layer_forward_moe_backward_moe_overlaping(
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
    checkpoint=False
):
    print(f"in transformer_layer_forward_moe_backward_moe_overlaping, start")
    if checkpoint:
        checkpoint_context = torch.no_grad()
    else:
        checkpoint_context = nullcontext()
    args = get_args()
    ep_group = parallel_state.get_expert_model_parallel_group()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    use_shared_experts = hasattr(fwd_layer.mlp, 'shared_experts') and fwd_layer.mlp.shared_experts is not None
    recomp_norm = getattr(args, 'recompute_norm', False)
    bwd_dispached_input, bwd_fc1_out, bwd_act_out, bwd_probs, bwd_indices, global_input_tokens_local_experts_indices = bwd_layer_graph.recompute_needed_tensors

    # Unperm2 Bwd
    # check if backward unpermutation alltoall is launched at bwd layer before
    if bwd_unperm_a2a_handle is None:
        run_graph_backward(bwd_layer_graph.unperm2_graph, bwd_layer_output_grad)
        # Async Unperm A2A
        _, unperm1_out_grad, bwd_unperm_a2a_handle = async_all_to_all(
            bwd_layer_graph.unperm_a2a_graph[1].grad,
            bwd_layer_graph.output_splits,
            bwd_layer_graph.input_splits,
            ep_group
        )
    else:
        unperm1_out_grad = bwd_layer_output_grad

    # # skip recompute
    # if args.moe_zero_memory == 'level0':
    #     ...

    with checkpoint_context:

        # Residual connection.
        detached_layer_input = detach_tensor(hidden_states)
        residual1 = detached_layer_input

        # input_layernorm + AttentionForward
        hidden_states = attention_forward(
            fwd_layer, detached_layer_input, residual1,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            recompute_norm=recomp_norm
        )

        attention_out, detached_attention_out = hidden_states, detach_tensor(hidden_states)

        # Residual connection.
        residual2 = detached_attention_out

        if recomp_norm:
            assert not recomp_norm, "not support recompute norm"
            pre_mlp_layernorm_output = fwd_layer.norm_ckpt2.checkpoint(fwd_layer.pre_mlp_layernorm, False, detached_attention_out)
        else:
            pre_mlp_layernorm_output = fwd_layer.pre_mlp_layernorm(detached_attention_out)
        # MLP.
        detached_mlp_input = detach_tensor(pre_mlp_layernorm_output)
        probs, routing_map = router_forward(fwd_layer, detached_mlp_input)
        if tp_size > 1 and use_shared_experts:
            # launch tp comm here and wait last aync comm finish
            _, shared_experts_input, shared_experts_allgather_handle = async_all_gather(
                detached_mlp_input, tp_group,
            )
            AG_SHARED_EXPERTS_INPUTS.append((shared_experts_input, shared_experts_allgather_handle))
        else:
            shared_experts_input, shared_experts_allgather_handle = detached_mlp_input, None

        # Token Permutation Forward
        probs_detached = detach_tensor(probs, checkpoint_forward=checkpoint)
        perm1_local_input_tokens, perm1_probs, tokens_per_expert = alltoall_token_perm1(fwd_layer.mlp.token_dispatcher, detached_mlp_input, probs_detached, routing_map)
    
    # # skip recompute
    # if args.moe_zero_memory == 'level0' or should_recompute_activation(bwd_layer_graph.layer.layer_number):
    #     ...
    
    # # skip recompute
    # if args.moe_zero_memory == 'level0':
    #     ...

    with checkpoint_context:
        # Async Perm A2A.
        _, perm1_local_input_tokens_a2a, perm1_local_input_tokens_a2a_handle = async_all_to_all(
            perm1_local_input_tokens,
            fwd_layer.mlp.token_dispatcher.output_splits,
            fwd_layer.mlp.token_dispatcher.input_splits,
            ep_group
        )
        _, perm1_probs_a2a, perm1_probs_a2a_handle = async_all_to_all(
            perm1_probs,
            fwd_layer.mlp.token_dispatcher.output_splits,
            fwd_layer.mlp.token_dispatcher.input_splits,
            ep_group
        )

    with checkpoint_context:
        shared_expert_output = None
        if use_shared_experts:
            if shared_experts_allgather_handle is not None:
                shared_experts_allgather_handle.wait()
                shared_experts_allgather_handle = None
            shared_expert_output = fwd_layer.mlp.shared_experts(detached_mlp_input)
            if tp_size > 1:
                # launch tp comm after permf a2a and wait until shared experts computation finish.
                share_experts_graph, shared_expert_output, rs_shared_experts_handle = async_reduce_scatter(
                    shared_expert_output, tp_group,
                )
            else:
                share_experts_graph = shared_expert_output
                rs_shared_experts_handle = None
    if recomp_norm:
        assert not recomp_norm, "not support recompute norm"
        fwd_layer.norm_ckpt2.discard_output()

    bwd_unperm_a2a_handle.wait()
    bwd_unperm_a2a_handle = None
    run_graph_backward(bwd_layer_graph.unperm1_graph, unperm1_out_grad)
    unperm1_out_grad.untyped_storage().resize_(0)
    WeightGradStore.start_decouple()
    run_graph_backward(bwd_layer_graph.grouped_mlp_graph, keep_grad=True)  # keep for dw
    WeightGradStore.end_decouple()
    run_graph_backward(bwd_layer_graph.perm2_graph, keep_graph=True)  # keep for dw
    run_graph_backward(bwd_layer_graph.perm2_append_graph, keep_graph=True)
    
    perm1_local_input_tokens_a2a_handle.wait()
    perm1_local_input_tokens_a2a_handle = None
    perm1_probs_a2a_handle.wait()
    perm1_probs_a2a_handle = None
    perm1_local_input_tokens.untyped_storage().resize_(0)
    perm1_probs.untyped_storage().resize_(0)
    
    _, perm1_out1_grad, bwd_perm_a2a_handle1 = async_all_to_all(
        bwd_layer_graph.perm_a2a_graph[1].grad,
        bwd_layer_graph.input_splits,
        bwd_layer_graph.output_splits,
        ep_group
    )

    _, perm1_out2_grad, bwd_perm_a2a_handle2 = async_all_to_all(
        bwd_layer_graph.perm_a2a_append_graph[1].grad,
        bwd_layer_graph.input_splits,
        bwd_layer_graph.output_splits,
        ep_group
    )

    # launch shared expert grad allgather here
    if tp_size > 1:
        _, backward_ag_shared, backward_ag_shared_handle = async_all_gather(
            bwd_layer_graph.shared_experts_graph[1].grad, tp_group,
        )
    else:
        backward_ag_shared = bwd_layer_graph.shared_experts_graph[1].grad
        backward_ag_shared_handle = None

    # Grouped MLP dw computation

    # # skip recompute
    # if args.moe_zero_memory == 'level0':
    #     ...

    WeightGradStore.pop()

    with checkpoint_context:
        detached_perm1_local_input_tokens_a2a = detach_tensor(perm1_local_input_tokens_a2a, checkpoint_forward=checkpoint)
        detached_perm1_probs_a2a = detach_tensor(perm1_probs_a2a, checkpoint_forward=checkpoint)
        # Token Perm2 Forward.
        dispatched_input, tokens_per_expert, permuted_probs = alltoall_token_perm2(fwd_layer.mlp.token_dispatcher, detached_perm1_local_input_tokens_a2a, detached_perm1_probs_a2a, tokens_per_expert)
        perm1_local_input_tokens_a2a.untyped_storage().resize_(0)
        perm1_probs_a2a.untyped_storage().resize_(0)

         # Grouped MLP Forward
        detached_dispatched_input = detach_tensor(dispatched_input, checkpoint_forward=checkpoint)
        detached_permuted_probs = detach_tensor(permuted_probs, checkpoint_forward=checkpoint)
        expert_output, mlp_bias = fwd_layer.mlp.experts(detached_dispatched_input, tokens_per_expert, detached_permuted_probs)
        
        # # skip recompute
        # if args.moe_zero_memory == 'level0':
        #     ...
        
        recompute_needed_tensors = [None, None, None, None, None, None]
        detached_expert_output = detach_tensor(expert_output, checkpoint_forward=checkpoint)
        detached_mlp_bias = detach_tensor(mlp_bias, checkpoint_forward=checkpoint)

        # Token Unpermutaion Forward
        unperm1_hidden_states = alltoall_token_unperm1(fwd_layer.mlp.token_dispatcher, detached_expert_output, detached_mlp_bias)
        expert_output.untyped_storage().resize_(0)
        if rs_shared_experts_handle is not None:
            rs_shared_experts_handle.wait()
            rs_shared_experts_handle = None
            share_experts_graph.untyped_storage().resize_(0)
        bwd_perm_a2a_handle1.wait()
        bwd_perm_a2a_handle1 = None
        bwd_perm_a2a_handle2.wait()
        bwd_perm_a2a_handle2 = None
    if backward_ag_shared_handle is not None:
        # ensure tp comm is not overlaped with alltoall comm
        backward_ag_shared_handle.wait()
        backward_ag_shared_handle = None
    # move shared experts backward before unpermF all2all to avoid tp comm colision.
    WeightGradStore.start_decouple()
    run_graph_backward(bwd_layer_graph.shared_experts_graph, backward_ag_shared, keep_grad=True)  # dw computation
    WeightGradStore.end_decouple()

    with checkpoint_context:
        # launch async all2all in the middle of attention graph backward
        _, unperm1_hidden_states_a2a, unperm1_hidden_states_a2a_handle = async_all_to_all(
            unperm1_hidden_states,
            fwd_layer.mlp.token_dispatcher.input_splits,
            fwd_layer.mlp.token_dispatcher.output_splits,
            ep_group
        )

    run_graph_backward(bwd_layer_graph.perm1_graph, perm1_out1_grad)
    run_graph_backward(bwd_layer_graph.perm1_append_graph, perm1_out2_grad)
    perm1_out1_grad.untyped_storage().resize_(0)
    perm1_out2_grad.untyped_storage().resize_(0)
    run_graph_backward(bwd_layer_graph.router_graph)
    run_graph_backward(bwd_layer_graph.pre_mlp_layernorm_graph, keep_graph=True)
    WeightGradStore.start_decouple()
    run_graph_backward(bwd_layer_graph.attn_graph, keep_grad=True)
    WeightGradStore.end_decouple()

    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        run_graph_backward(next_bwd_layer_graph.unperm2_graph, bwd_layer_graph.layer_input.grad, keep_graph=True)

    unperm1_hidden_states_a2a_handle.wait()
    unperm1_hidden_states_a2a_handle = None
    unperm1_hidden_states.untyped_storage().resize_(0)

    next_layer_output_grad, next_bwd_unperm_a2a_handle = bwd_layer_graph.layer_input.grad, None
    if next_bwd_layer_graph is not None and getattr(next_bwd_layer_graph, 'is_moe_layer', False):
        _, next_layer_output_grad, next_bwd_unperm_a2a_handle = async_all_to_all(
            next_bwd_layer_graph.unperm_a2a_graph[1].grad,
            next_bwd_layer_graph.output_splits,
            next_bwd_layer_graph.input_splits,
            ep_group
        )
    with checkpoint_context:
        detached_unperm1_hidden_states_a2a = detach_tensor(unperm1_hidden_states_a2a, checkpoint_forward=checkpoint)
        route_expert_output, _ = alltoall_token_unperm2(fwd_layer.mlp.token_dispatcher, detached_unperm1_hidden_states_a2a)

        if hasattr(fwd_layer.mlp, 'shared_experts') and fwd_layer.mlp.shared_experts is not None:
            detached_shared_expert_output = detach_tensor(shared_expert_output)
            mlp_output = route_expert_output + detached_shared_expert_output
            shared_expert_output.untyped_storage().resize_(0)
        else:
            detached_shared_expert_output = None
            share_experts_graph = None
            mlp_output = route_expert_output

        if recomp_norm:
            assert not recomp_norm, "not support recompute norm"
            mlp_output.register_hook(fwd_layer.norm_ckpt2.recompute)


        with fwd_layer.bias_dropout_add_exec_handler():
            hidden_states = fwd_layer.mlp_bda(fwd_layer.training, fwd_layer.config.bias_dropout_fusion)(
                (mlp_output, None), residual2, fwd_layer.hidden_dropout
            )

    # Jit compiled function creates 'view' tensor. This tensor
    # potentially gets saved in the MPU checkpoint function context,
    # which rejects view tensors. While making a viewless tensor here
    # won't result in memory savings (like the data loader, or
    # p2p_communication), it serves to document the origin of this
    # 'view' tensor.
    output = make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )

    # handle fwd p2p communication
    next_iter_input_tensor, fwd_p2p_handles = None, None
    fwd_pp_comm_params = pp_comm_params
    if is_p2p_comm_needed(fwd_pp_comm_params):
        next_iter_input_tensor, fwd_p2p_handles = p2p_comm_helper(fwd_pp_comm_params, output)

    # handle bwd p2p communication
    next_iter_output_tensor_grad, bwd_p2p_handles = None, None
    if is_p2p_comm_needed(bwd_pp_comm_params):
        next_iter_output_tensor_grad, bwd_p2p_handles = p2p_comm_helper(bwd_pp_comm_params, bwd_layer_graph.layer_input.grad)

    WeightGradStore.pop()

    saved_tensors = (
        (attention_out, detached_attention_out),
        (pre_mlp_layernorm_output, detached_mlp_input),
        (probs, probs_detached),
        (perm1_local_input_tokens, None),  # perm1 graph
        (perm1_probs, None), # perm1 append graph
        (None, detached_perm1_local_input_tokens_a2a),
        (None, detached_perm1_probs_a2a),
        (dispatched_input, detached_dispatched_input),  # perm2 graph
        (permuted_probs, detached_permuted_probs), # perm2 append graph
        (expert_output, detached_expert_output),  # grouped mlp graph
        (unperm1_hidden_states, None),  # unperm1 graph
        (None, detached_unperm1_hidden_states_a2a),
        (output, None),  # unperm2 graph
        (share_experts_graph, detached_shared_expert_output),
        detached_layer_input
    )

    graph = LayerGraph(
        saved_tensors, recompute_needed_tensors,
        fwd_layer.mlp.token_dispatcher.input_splits, fwd_layer.mlp.token_dispatcher.output_splits, fwd_layer,
        checkpointed=checkpoint
    )

    for tensor in bwd_layer_graph.recompute_needed_tensors:
        if tensor is not None:
            tensor.untyped_storage().resize_(0)

    print(f"in transformer_layer_forward_moe_backward_moe_overlaping, return")
    return (output, context, graph,
            (next_layer_output_grad, next_bwd_unperm_a2a_handle),
            P2PCommOutput(next_iter_input_tensor, next_iter_output_tensor_grad, fwd_p2p_handles, bwd_p2p_handles, bwd_layer_graph.layer_input.grad))
