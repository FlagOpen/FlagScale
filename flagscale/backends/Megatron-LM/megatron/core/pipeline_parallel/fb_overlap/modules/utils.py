import torch
import torch.distributed as dist
from torch import Tensor
from torch.autograd.variable import Variable
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.parallel_state import get_global_memory_buffer, get_tensor_model_parallel_rank
from typing import Dict, Literal, Optional, Tuple, Union, List


AG_SHARED_EXPERTS_INPUTS = []

def async_all_gather(input_, group, event=None, is_use_get_global_memory_buffer=False, last_dim=False, stream=None):
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_, input_, None
    if last_dim:
        rank = get_tensor_model_parallel_rank()
        ag_out = [torch.empty_like(input_) for _ in range(world_size)]
        ag_out[rank] = input_
    else:
        dim_size = list(input_.size())
        new_dim_size = dim_size[0] * world_size
        dim_size[0] = new_dim_size

        if is_use_get_global_memory_buffer:
            ag_out = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            ag_out = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    
    if last_dim:
        handle = torch.distributed.all_gather(ag_out, input_.contiguous(), group=group, async_op=True)
    else:
        handle = torch.distributed._all_gather_base(
            ag_out, input_.contiguous(), group=group, async_op=True
        )
    return input_, ag_out, handle


def async_reduce_scatter(input_, group, event=None, stream=None, is_use_get_global_memory_buffer=False):
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_, input_, None
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] // world_size
    if is_use_get_global_memory_buffer:
        rs_out = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
    else:
        rs_out = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    
    handle = torch.distributed._reduce_scatter_base(
        rs_out, input_.contiguous(), group=group, async_op=True
    )
    return input_, rs_out, handle


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

    handle = dist.all_to_all_single(
        a2a_out,
        input_.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=True
    )
    print(f"return handle here")
    return input_, a2a_out, handle

def detach_tensor(tensor, checkpoint_forward=False):
    if checkpoint_forward:
        return tensor
    if tensor is None:
        return None
    detached_tensor = tensor.detach()
    detached_tensor.requires_grad = True
    return detached_tensor


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


class NoopLayerGraph:
    def __init__(self, layer_input, layer_output, layer, checkpointed=False):
        self.layer_input = layer_input
        if not checkpointed:
            self.unperm2_graph = (layer_output, None)
        else:
            self.unperm2_graph = (None, None)
        self.checkpointed = checkpointed
        self.layer = layer

    def record_layer_inputs(self, *args):
        self.layer_inputs = args


class LayerGraph:
    def __init__(self, saved_graph_and_graph_inputs, recompute_needed_tensors, input_splits, output_splits, layer, checkpointed=False):
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
    def __init__(self, input_tensor=None, output_tensor_grad=None, fwd_wait_handles=None, bwd_wait_handles=None, input_tensor_grad=None):
        self.input_tensor = input_tensor
        self.fwd_wait_handles = fwd_wait_handles
        self.output_tensor_grad = output_tensor_grad
        self.bwd_wait_handles = bwd_wait_handles
        self.input_tensor_grad = input_tensor_grad


def is_p2p_comm_needed(pp_comm_params: P2PCommParams):
    return pp_comm_params is not None and \
           (pp_comm_params.send_next or pp_comm_params.send_prev or pp_comm_params.recv_next or pp_comm_params.recv_prev)


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
        config=comm_params.config
    )

    if comm_params.recv_next:
        return tensor_recv_next, p2p_handles
    elif comm_params.recv_prev:
        return tensor_recv_prev, p2p_handles
    else:
        return None, p2p_handles

