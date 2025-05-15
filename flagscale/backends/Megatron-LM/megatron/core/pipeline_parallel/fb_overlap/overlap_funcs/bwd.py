#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.core.transformer.moe.moe_utils import permute
from megatron.core.pipeline_parallel.fb_overlap.modules.utils import async_all_gather, async_all_to_all
from ..modules.weight_grad_store import WeightGradStore
from ..modules.utils import (
    run_graph_backward,
    turn_attention_delay_wgrad_compute,
    call_attention_backward_dw,
    turn_shared_experts_delay_wgrad_compute,
    call_shared_experts_backward_dw,
    turn_experts_delay_wgrad_compute,
    call_experts_backward_dw,
    turn_dense_mlp_delay_wgrad_compute,
    call_dense_mlp_backward_dw,
)


def transformer_layer_backward_moe(
    layer_output_grad,
    layer_graph
):
    # print(f"in transformer_layer_backward_moe, start")
    self = layer_graph
    args = get_args()
    dispached_input, fc1_out, act_out, probs, indices, global_input_tokens_local_experts_indices = self.recompute_needed_tensors
    ep_group = parallel_state.get_expert_model_parallel_group()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_size > 1:
        shared_expert_grad = layer_output_grad if layer_output_grad is not None else self.unperm2_graph[1].grad
        _, backward_ag_shared, backward_ag_shared_handle = async_all_gather(
            shared_expert_grad, parallel_state.get_tensor_model_parallel_group()
        )
    else:
        backward_ag_shared = layer_output_grad if layer_output_grad is not None else self.unperm2_graph[1].grad
        backward_ag_shared_handle = None

    run_graph_backward(self.unperm2_graph, layer_output_grad, keep_grad=True)
    if backward_ag_shared_handle is not None:
        backward_ag_shared_handle.wait()
        backward_ag_shared_handle = None
        if layer_output_grad is not None:
            layer_output_grad.untyped_storage().resize_(0)
    _, unperm1_out_grad, bwd_unperm_a2a_handle = async_all_to_all(
        self.unperm_a2a_graph[1].grad,
        self.output_splits,
        self.input_splits,
        ep_group
    )
    # overlap alltoall by shared experts backward
    if self.shared_experts_graph[0] is not None:
        WeightGradStore.start_decouple()
        turn_shared_experts_delay_wgrad_compute(self, enable=True)
        run_graph_backward(self.shared_experts_graph, backward_ag_shared)
        WeightGradStore.end_decouple()
        WeightGradStore.pop()
        call_shared_experts_backward_dw(self)
        turn_shared_experts_delay_wgrad_compute(self, enable=False)
    
    # # skip recompute
    # if get_args().moe_zero_memory == 'level0' or should_recompute_activation(self.layer.layer_number):
    #   ...

    bwd_unperm_a2a_handle.wait()
    bwd_unperm_a2a_handle = None

    # # skip recompute
    # if get_args().moe_zero_memory == 'level0':
    #    ...

    run_graph_backward(self.unperm1_graph, unperm1_out_grad)

    WeightGradStore.start_decouple()
    turn_experts_delay_wgrad_compute(self, enable=True)
    run_graph_backward(self.grouped_mlp_graph, keep_grad=True)  # keep for dw commputation
    WeightGradStore.end_decouple()

    run_graph_backward(self.perm2_graph, keep_graph=True)  # keep for dw commutation
    run_graph_backward(self.perm2_append_graph, keep_graph=True)
    
    # # remove recompute
    # if get_args().moe_zero_memory == 'level0':
        # perm_a2a_handle.wait()
        # perm_a2a_handle = None

    _, perm1_out1_grad, bwd_perm_a2a_handle1 = async_all_to_all(
        self.perm_a2a_graph[1].grad,
        self.input_splits,
        self.output_splits,
        ep_group
    )

    _, perm1_out2_grad, bwd_perm_a2a_handle2 = async_all_to_all(
        self.perm_a2a_append_graph[1].grad,
        self.input_splits,
        self.output_splits,
        ep_group
    )

    # # skip recompute
    # if get_args().moe_zero_memory == 'level0':
    #     ...

    # dw computation
    WeightGradStore.pop()
    call_experts_backward_dw(self)
    turn_experts_delay_wgrad_compute(self, enable=False)
    
    bwd_perm_a2a_handle1.wait()
    bwd_perm_a2a_handle1 = None
    bwd_perm_a2a_handle2.wait()
    bwd_perm_a2a_handle2 = None

    run_graph_backward(self.perm1_graph, perm1_out1_grad)
    run_graph_backward(self.perm1_append_graph, perm1_out2_grad)
    perm1_out1_grad.untyped_storage().resize_(0)
    perm1_out2_grad.untyped_storage().resize_(0)
    run_graph_backward(self.router_graph)
    run_graph_backward(self.pre_mlp_layernorm_graph)

    WeightGradStore.start_decouple()
    turn_attention_delay_wgrad_compute(self, enable=True)
    run_graph_backward(self.attn_graph)
    WeightGradStore.end_decouple()
    WeightGradStore.pop()
    call_attention_backward_dw(self)
    turn_attention_delay_wgrad_compute(self, enable=False)

    self.recompute_needed_tensors = [None for _ in range(len(self.recompute_needed_tensors))]

    # print(f"in transformer_layer_backward_moe, return")
    return self.layer_input.grad


def transformer_layer_backward_dense(layer_output_grad, layer_graph):
    # print(f"in transformer_layer_backward_dense, start")
    
    WeightGradStore.start_decouple()
    turn_dense_mlp_delay_wgrad_compute(layer_graph, enable=True)
    run_graph_backward(layer_graph.unperm2_graph, layer_output_grad)
    run_graph_backward(layer_graph.pre_mlp_layernorm_graph)
    WeightGradStore.end_decouple()
    WeightGradStore.pop()
    call_dense_mlp_backward_dw(layer_graph)
    turn_dense_mlp_delay_wgrad_compute(layer_graph, enable=False)


    WeightGradStore.start_decouple()
    turn_attention_delay_wgrad_compute(layer_graph, enable=True)
    run_graph_backward(layer_graph.attn_graph)
    WeightGradStore.end_decouple()
    WeightGradStore.pop()
    call_attention_backward_dw(layer_graph)
    turn_attention_delay_wgrad_compute(layer_graph, enable=False)

    # print(f"in transformer_layer_backward_dense, return")
    return layer_graph.layer_input.grad


def transformer_layer_backward_noop(layer_output_grad, layer_graph):
    # print(f"in transformer_layer_backward_noop, start")
    run_graph_backward(layer_graph.unperm2_graph, layer_output_grad, keep_grad=True)
    # print(f"in transformer_layer_backward_noop, return")
    return layer_graph.layer_input.grad

