from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist

from torch import Tensor
from torch.autograd.variable import Variable

from megatron.core.parallel_state import get_global_memory_buffer, get_tensor_model_parallel_rank
from megatron.core.pipeline_parallel import p2p_communication

COMM_STREAM = None


def async_all_to_all(input_, output_split_sizes, input_split_sizes, group, event=None, stream=None):
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_, input_, None
    if output_split_sizes is None:
        # Equal split (all2all)
        a2a_out = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.cuda.current_device(),
        )

    if event or stream:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch.cuda.Stream(device=torch.cuda.current_device())
        with torch.cuda.stream(COMM_STREAM):
            if event:
                event.wait()
            if stream:
                COMM_STREAM.wait_stream(stream)
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )
    else:
        handle = dist.all_to_all_single(
            a2a_out,
            input_.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
    return input_, a2a_out, handle


def detach_tensor(tensor, checkpoint_forward=False):
    if checkpoint_forward:
        return tensor
    if tensor is None:
        return None
    detached_tensor = tensor.detach()
    detached_tensor.requires_grad = True
    return detached_tensor


def turn_attention_delay_wgrad_compute(bwd_layer_graph, enable=False):
    attention_layer = bwd_layer_graph.layer.self_attention
    has_linear_qkv = any("linear_qkv" in name for name, _ in attention_layer.named_modules())
    if enable:
        attention_layer.linear_proj.wgrad_store.enable_delay_wgrad_compute()
        if has_linear_qkv:  # for self_attention
            attention_layer.linear_qkv.wgrad_store.enable_delay_wgrad_compute()
        else:  # for multi_latent_attention
            attention_layer.linear_kv_up_proj.wgrad_store.enable_delay_wgrad_compute()
            attention_layer.linear_kv_down_proj.wgrad_store.enable_delay_wgrad_compute()
            attention_layer.linear_q_proj.wgrad_store.enable_delay_wgrad_compute()

    else:
        attention_layer.linear_proj.wgrad_store.disable_delay_wgrad_compute()
        if has_linear_qkv:
            attention_layer.linear_qkv.wgrad_store.disable_delay_wgrad_compute()
        else:
            attention_layer.linear_kv_up_proj.wgrad_store.disable_delay_wgrad_compute()
            attention_layer.linear_kv_down_proj.wgrad_store.disable_delay_wgrad_compute()
            attention_layer.linear_q_proj.wgrad_store.disable_delay_wgrad_compute()


def call_attention_backward_dw(bwd_layer_graph):
    attention_layer = bwd_layer_graph.layer.self_attention
    has_linear_qkv = any("linear_qkv" in name for name, _ in attention_layer.named_modules())

    attention_layer.linear_proj.backward_dw()
    if has_linear_qkv:
        attention_layer.linear_qkv.backward_dw()
    else:
        attention_layer.linear_kv_up_proj.backward_dw()
        attention_layer.linear_kv_down_proj.backward_dw()
        attention_layer.linear_q_proj.backward_dw()


def turn_shared_experts_delay_wgrad_compute(bwd_layer_graph, enable=False):
    shared_experts_layer = bwd_layer_graph.layer.mlp.shared_experts
    if enable:
        shared_experts_layer.linear_fc2.wgrad_store.enable_delay_wgrad_compute()
        shared_experts_layer.linear_fc1.wgrad_store.enable_delay_wgrad_compute()
    else:
        shared_experts_layer.linear_fc2.wgrad_store.disable_delay_wgrad_compute()
        shared_experts_layer.linear_fc1.wgrad_store.disable_delay_wgrad_compute()


def call_shared_experts_backward_dw(bwd_layer_graph):
    shared_experts_layer = bwd_layer_graph.layer.mlp.shared_experts
    shared_experts_layer.linear_fc2.backward_dw()
    shared_experts_layer.linear_fc1.backward_dw()


def turn_experts_delay_wgrad_compute(bwd_layer_graph, enable=False):
    experts_layer = bwd_layer_graph.layer.mlp.experts
    if enable:
        experts_layer.linear_fc2.wgrad_store.enable_delay_wgrad_compute()
        experts_layer.linear_fc1.wgrad_store.enable_delay_wgrad_compute()
    else:
        experts_layer.linear_fc2.wgrad_store.disable_delay_wgrad_compute()
        experts_layer.linear_fc1.wgrad_store.disable_delay_wgrad_compute()


def call_experts_backward_dw(bwd_layer_graph):
    experts_layer = bwd_layer_graph.layer.mlp.experts
    experts_layer.linear_fc2.backward_dw()
    experts_layer.linear_fc1.backward_dw()


def turn_dense_mlp_delay_wgrad_compute(bwd_layer_graph, enable=False):
    dense_mlp_layer = bwd_layer_graph.layer.mlp
    if enable:
        dense_mlp_layer.linear_fc2.wgrad_store.enable_delay_wgrad_compute()
        dense_mlp_layer.linear_fc1.wgrad_store.enable_delay_wgrad_compute()
    else:
        dense_mlp_layer.linear_fc2.wgrad_store.disable_delay_wgrad_compute()
        dense_mlp_layer.linear_fc1.wgrad_store.disable_delay_wgrad_compute()


def call_dense_mlp_backward_dw(bwd_layer_graph):
    dense_mlp_layer = bwd_layer_graph.layer.mlp
    dense_mlp_layer.linear_fc2.backward_dw()
    dense_mlp_layer.linear_fc1.backward_dw()


def run_graph_backward(graph, output_tensor_grad=None, keep_graph=False, keep_grad=False):
    grad_tensor = output_tensor_grad
    if output_tensor_grad is None and graph[1] is not None and graph[1].grad is not None:
        grad_tensor = graph[1].grad
    Variable._execution_engine.run_backward(
        tensors=(graph[0],),
        grad_tensors=(grad_tensor,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )

    if not keep_graph:
        graph[0].untyped_storage().resize_(0)
    if not keep_grad:
        grad_tensor.untyped_storage().resize_(0)


class LayerGraph:
    def __init__(
        self,
        saved_graph_and_graph_inputs,
        recompute_needed_tensors,
        input_splits,
        output_splits,
        layer,
        checkpointed=False,
    ):
        if not checkpointed:
            self.attn_graph = saved_graph_and_graph_inputs[0]
            self.pre_mlp_layernorm_graph = saved_graph_and_graph_inputs[1]
            self.router_graph = saved_graph_and_graph_inputs[2]
            self.perm1_graph = saved_graph_and_graph_inputs[3]
            self.perm1_append_graph = saved_graph_and_graph_inputs[4]
            self.perm_a2a_graph = saved_graph_and_graph_inputs[5]
            self.perm_a2a_append_graph = saved_graph_and_graph_inputs[6]
            self.perm2_graph = saved_graph_and_graph_inputs[7]
            self.perm2_append_graph = saved_graph_and_graph_inputs[8]
            self.grouped_mlp_graph = saved_graph_and_graph_inputs[9]
            self.unperm1_graph = saved_graph_and_graph_inputs[10]
            self.unperm_a2a_graph = saved_graph_and_graph_inputs[11]
            self.unperm2_graph = saved_graph_and_graph_inputs[12]
            self.shared_experts_graph = saved_graph_and_graph_inputs[13]
        else:
            self.unperm2_graph = (None, None)

        self.layer_input = saved_graph_and_graph_inputs[-1]
        self.recompute_needed_tensors = recompute_needed_tensors
        self.input_splits = input_splits
        self.output_splits = output_splits
        self.checkpointed = checkpointed
        self.layer = layer
        self.is_moe_layer = hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts')

    def record_layer_inputs(self, *args):
        self.layer_inputs = args


class ModelGraph:
    def __init__(
        self,
        layer_graphs: List[LayerGraph],
        block_output,
        preprocess_graph: Tensor = None,
        preprocess_detached_output: Tensor = None,
    ):
        self.preprocess_graph = (preprocess_graph, preprocess_detached_output)
        self.layer_graphs = layer_graphs
        self.block_output = block_output


class P2PCommParams:
    tensor_shape = None
    config = None

    def __init__(self, send_next=False, send_prev=False, recv_next=False, recv_prev=False):
        self.send_next = send_next
        self.send_prev = send_prev
        self.recv_next = recv_next
        self.recv_prev = recv_prev

    def __str__(self):
        return f'send next:{self.send_next} send_prev:{self.send_prev} recv_next:{self.recv_next} recv_prev:{self.recv_prev}'


class P2PCommOutput:
    def __init__(
        self,
        input_tensor=None,
        output_tensor_grad=None,
        fwd_wait_handles=None,
        bwd_wait_handles=None,
        input_tensor_grad=None,
    ):
        self.input_tensor = input_tensor
        self.fwd_wait_handles = fwd_wait_handles
        self.output_tensor_grad = output_tensor_grad
        self.bwd_wait_handles = bwd_wait_handles
        self.input_tensor_grad = input_tensor_grad


def is_p2p_comm_needed(pp_comm_params: P2PCommParams):
    return pp_comm_params is not None and (
        pp_comm_params.send_next
        or pp_comm_params.send_prev
        or pp_comm_params.recv_next
        or pp_comm_params.recv_prev
    )


def p2p_comm_helper(comm_params: P2PCommParams, tensor_tosend):
    assert not (comm_params.send_next and comm_params.send_prev)
    assert not (comm_params.recv_next and comm_params.recv_prev)
    tensor_send_next = None
    if comm_params.send_next:
        tensor_send_next = tensor_tosend
    tensor_send_prev = None
    if comm_params.send_prev:
        tensor_send_prev = tensor_tosend
    tensor_recv_prev, tensor_recv_next, p2p_handles = p2p_communication._communicate(
        tensor_send_next=tensor_send_next,
        tensor_send_prev=tensor_send_prev,
        recv_prev=comm_params.recv_prev,
        recv_next=comm_params.recv_next,
        tensor_shape=comm_params.tensor_shape,
        wait_on_reqs=False,
        config=comm_params.config,
    )

    if comm_params.recv_next:
        return tensor_recv_next, p2p_handles
    elif comm_params.recv_prev:
        return tensor_recv_prev, p2p_handles
    else:
        return None, p2p_handles
