# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import os
import warnings
from typing import Any, Callable, List, Optional

import torch
import torch.distributed
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.parameter import Parameter
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    linear_with_grad_accumulation_and_async_allreduce,
    linear_with_frozen_weight
)
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
    _reduce_scatter_along_first_dim,
    _gather_along_first_dim
)
from megatron.core.tensor_parallel.utils import VocabUtility, divide, split_tensor_along_last_dim
from megatron.core.utils import (
    make_tp_sharded_tensor_for_checkpoint,
    prepare_input_tensors_for_wgrad_compute
)
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.training import get_args
from megatron.core.pipeline_parallel.fb_overlap.modules.weight_grad_store import WeightGradStore

### All linear layer are using TransformerEngine rather than Megatron Linear
### So we do not need to modify here

# def linear_backward_wgrad_detach(ctx, grad_output):
#     input_, weight = ctx.saved_tensors
#     use_bias = ctx.use_bias
#     grad_output_buffer = ctx.grad_output_buffer
#     wgrad_deferral_limit = ctx.wgrad_deferral_limit

#     wgrad_compute = True
#     if grad_output_buffer is not None:
#         if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
#             grad_output_buffer.append(grad_output)
#             wgrad_compute = False

#     if wgrad_compute:
#         if ctx.sequence_parallel and not WeightGradStore.is_decoupleBlock:
#             world_size = get_tensor_model_parallel_world_size()
#             dim_size = list(input_.size())
#             dim_size[0] = dim_size[0] * world_size

#             all_gather_buffer = get_global_memory_buffer().get_tensor(
#                 dim_size, input_.dtype, "mpu"
#             )
#             handle = torch.distributed._all_gather_base(
#                 all_gather_buffer, input_, group=get_tensor_model_parallel_group(), async_op=True
#             )

#             # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
#             # gather is scheduled before the input gradient computation
#             total_input = all_gather_buffer
#         else:
#             total_input = input_
#     grad_input = grad_output.matmul(weight)

#     if ctx.sequence_parallel and wgrad_compute and not WeightGradStore.is_decoupleBlock:
#         handle.wait()

#     if wgrad_compute and not WeightGradStore.is_decoupleBlock:
#         grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
#             grad_output, total_input
#         )

#     if ctx.allreduce_dgrad:
#         # Asynchronous all-reduce
#         handle = torch.distributed.all_reduce(
#             grad_input, group=get_tensor_model_parallel_group(), async_op=True
#         )
#         # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
#         # all-reduce is scheduled before the weight gradient computation

#     if ctx.sequence_parallel:
#         assert not ctx.allreduce_dgrad
#         dim_size = list(input_.size())
#         sub_grad_input = torch.empty(
#             dim_size, dtype=input_.dtype, device=torch.cuda.current_device(), requires_grad=False
#         )
#         # reduce_scatter
#         handle = torch.distributed._reduce_scatter_base(
#             sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
#         )
#         # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
#         # reduce scatter is scheduled before the weight gradient computation


#     if WeightGradStore.is_decoupleBlock:
#         # TODO: remove clone under MLA setting
#         WeightGradStore.put(
#             total_input.clone().detach(),
#             grad_output.clone().detach(),
#             weight,
#             ctx.sequence_parallel,
#             in_row=not ctx.sequence_parallel
#         )
#         if hasattr(weight, 'grad_added_to_main_grad') and get_args().overlap_grad_reduce:
#             weight.skip_grad_accum = True
#         grad_weight = None
#     else:
#         if ctx.gradient_accumulation_fusion:
#             if wgrad_compute:
#                 if weight.main_grad.dtype == torch.float32:
#                     from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32
#                     npu_matmul_add_fp32(total_input, grad_output, weight.main_grad)
#                 elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
#                     raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

#             if hasattr(weight, 'grad_added_to_main_grad'):
#                 # When overlap_grad_reduce is True, need to ensure that backward hooks
#                 # are all run on the main backprop thread to prevent deadlocks. Setup
#                 # dummy grad_weight tensor to prevent backward hooks from being run
#                 # in a background thread.
#                 if getattr(weight, 'zero_out_wgrad', False):
#                     grad_weight = torch.zeros(
#                         weight.main_grad.shape,
#                         dtype=input_.dtype,
#                         device=torch.cuda.current_device(),
#                         requires_grad=False,
#                     )
#                 else:
#                     grad_weight = torch.empty(
#                         weight.main_grad.shape,
#                         dtype=input_.dtype,
#                         device=torch.cuda.current_device(),
#                         requires_grad=False,
#                     )
#                 weight.grad_added_to_main_grad = True
#             else:
#                 grad_weight = None
#         else:
#             grad_weight = grad_output.t().matmul(total_input)
#     grad_bias = grad_output.sum(dim=0) if use_bias else None

#     if ctx.sequence_parallel:
#         handle.wait()
#         # Need to return None's as gradient has to flow for all the input arguments
#         # provided during forward
#         return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None

#     if ctx.allreduce_dgrad:
#         handle.wait()

#     return grad_input, grad_weight, grad_bias, None, None, None, None, None


# class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
#     """See linear_with_grad_accumulation_and_async_allreduce"""

#     @staticmethod
#     @custom_fwd
#     def forward(
#         ctx,
#         input,
#         weight,
#         bias,
#         gradient_accumulation_fusion,
#         async_grad_allreduce,
#         sequence_parallel,
#         grad_output_buffer,
#         shared_expert,
#     ):
#         ctx.save_for_backward(input, weight)
#         ctx.use_bias = bias is not None
#         ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
#         ctx.async_grad_allreduce = async_grad_allreduce
#         ctx.sequence_parallel = sequence_parallel
#         ctx.grad_output_buffer = grad_output_buffer
#         ctx.shared_expert = shared_expert
#         if sequence_parallel:
#             if shared_expert:
#                 from mindspeed.core.transformer.moe.moe_utils import AG_SHARED_EXPERTS_INPUTS
#                 ag_shared_experts_inputs = AG_SHARED_EXPERTS_INPUTS.pop(0)
#                 if isinstance(ag_shared_experts_inputs, tuple):
#                     ag_shared_experts_inputs, handle = ag_shared_experts_inputs
#                     handle.wait()
#                 total_input = ag_shared_experts_inputs
#             else:
#                 world_size = get_tensor_model_parallel_world_size()
#                 dim_size = list(input.size())
#                 dim_size[0] = dim_size[0] * world_size

#                 all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
#                 torch.distributed._all_gather_base(
#                     all_gather_buffer, input, group=get_tensor_model_parallel_group()
#                 )
#                 total_input = all_gather_buffer
#         else:
#             total_input = input

#         output = torch.matmul(total_input, weight.t())

#         if bias is not None:
#             output = output + bias
#         return output

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, grad_output):
#         input, weight = ctx.saved_tensors
#         use_bias = ctx.use_bias
#         grad_output_buffer = ctx.grad_output_buffer

#         wgrad_compute = True
#         if grad_output_buffer is not None:
#             grad_output_buffer.append(grad_output)
#             wgrad_compute = False

#         if wgrad_compute:
#             if ctx.sequence_parallel and not WeightGradStore.is_decoupleBlock:
#                 world_size = get_tensor_model_parallel_world_size()
#                 dim_size = list(input.size())
#                 dim_size[0] = dim_size[0] * world_size

#                 all_gather_buffer = get_global_memory_buffer().get_tensor(
#                     dim_size, input.dtype, "mpu"
#                 )
#                 handle = torch.distributed._all_gather_base(
#                     all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
#                 )

#                 # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
#                 # gather is scheduled before the input gradient computation
#                 total_input = all_gather_buffer
#             else:
#                 total_input = input

#         grad_input = grad_output.matmul(weight)

#         if ctx.sequence_parallel and wgrad_compute and not WeightGradStore.is_decoupleBlock:
#             handle.wait()

#         if wgrad_compute and not WeightGradStore.is_decoupleBlock:
#             grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
#                 grad_output, total_input
#             )

#         if ctx.async_grad_allreduce:
#             # Asynchronous all-reduce
#             handle = torch.distributed.all_reduce(
#                 grad_input, group=get_tensor_model_parallel_group(), async_op=True
#             )
#             # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
#             # all-reduce is scheduled before the weight gradient computation

#         if ctx.sequence_parallel:
#             assert not ctx.async_grad_allreduce
#             dim_size = list(input.size())
#             sub_grad_input = torch.empty(
#                 dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
#             )
#             # reduce_scatter
#             handle = torch.distributed._reduce_scatter_base(
#                 sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
#             )
#             # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
#             # reduce scatter is scheduled before the weight gradient computation

#         if WeightGradStore.is_decoupleBlock:
#             # TODO: remove clone under MLA setting
#             WeightGradStore.put(
#                 total_input.clone().detach(),
#                 grad_output.clone().detach(),
#                 weight,
#                 ctx.sequence_parallel,
#                 in_row=not ctx.sequence_parallel
#             )
#             if hasattr(weight, 'grad_added_to_main_grad') and get_args().overlap_grad_reduce:
#                 weight.skip_grad_accum = True
#             grad_weight = None
#         else:
#             if ctx.gradient_accumulation_fusion:
#                 if wgrad_compute:
#                     if weight.main_grad.dtype == torch.float32:
#                         from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32
#                         npu_matmul_add_fp32(total_input, grad_output, weight.main_grad)
#                     elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
#                         raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

#                 if hasattr(weight, 'grad_added_to_main_grad'):
#                     # When overlap_grad_reduce is True, need to ensure that backward hooks
#                     # are all run on the main backprop thread to prevent deadlocks. Setup
#                     # dummy grad_weight tensor to prevent backward hooks from being run
#                     # in a background thread.
#                     if getattr(weight, 'zero_out_wgrad', False):
#                         grad_weight = torch.zeros(
#                             weight.main_grad.shape,
#                             dtype=input.dtype,
#                             device=torch.cuda.current_device(),
#                             requires_grad=False,
#                         )
#                     else:
#                         grad_weight = torch.empty(
#                             weight.main_grad.shape,
#                             dtype=input.dtype,
#                             device=torch.cuda.current_device(),
#                             requires_grad=False,
#                         )
#                     weight.grad_added_to_main_grad = True
#                 else:
#                     grad_weight = None
#             else:
#                 grad_weight = grad_output.t().matmul(total_input)
#         grad_bias = grad_output.sum(dim=0) if use_bias else None

#         if ctx.sequence_parallel:
#             handle.wait()
#             # Need to return None's as gradient has to flow for all the input arguments
#             # provided during forward
#             return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None

#         if ctx.async_grad_allreduce:
#             handle.wait()

#         return grad_input, grad_weight, grad_bias, None, None, None, None, None


# def linear_with_grad_accumulation_and_async_allreduce(
#     input: torch.Tensor,
#     weight: torch.Tensor,
#     bias: Optional[torch.Tensor],
#     gradient_accumulation_fusion: bool,
#     async_grad_allreduce: bool,
#     sequence_parallel: bool,
#     grad_output_buffer: Optional[List[torch.Tensor]] = None,
#     shared_expert: bool = False
# ) -> torch.Tensor:
#     args = [
#         input,
#         weight,
#         bias,
#         gradient_accumulation_fusion,
#         async_grad_allreduce,
#         sequence_parallel,
#         grad_output_buffer,
#         shared_expert,
#     ]

#     if not linear_with_grad_accumulation_and_async_allreduce.warned:
#         if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
#             if sequence_parallel:
#                 warnings.warn(
#                     "When using sequence parallelism it is recommended to set the "
#                     "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
#                     "maximum speedup"
#                 )
#                 linear_with_grad_accumulation_and_async_allreduce.warned = True

#             if async_grad_allreduce:
#                 warnings.warn(
#                     "When using async grad allreduce it is recommended to set the "
#                     "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
#                     "maximum speedup"
#                 )
#                 linear_with_grad_accumulation_and_async_allreduce.warned = True

#     return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)


# linear_with_grad_accumulation_and_async_allreduce.warned = False


# class ColumnParallelLinear(torch.nn.Module):

#     def __init__(
#         self,
#         input_size,
#         output_size,
#         *,
#         config: ModelParallelConfig,
#         init_method: Callable,
#         bias=True,
#         gather_output=False,
#         stride=1,
#         keep_master_weight_for_test=False,
#         skip_bias_add=False,
#         skip_weight_param_allocation: bool = False,
#         embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
#         grad_output_buffer: Optional[List[torch.Tensor]] = None,
#         is_expert: bool = False,
#         tp_comm_buffer_name: str = None,  # Not used
#         shared_expert: bool = False
#     ):
#         super(ColumnParallelLinear, self).__init__()

#         # Keep input parameters
#         self.input_size = input_size
#         self.output_size = output_size
#         self.gather_output = gather_output
#         # Divide the weight matrix along the last dimension.
#         world_size = get_tensor_model_parallel_world_size()
#         self.output_size_per_partition = divide(output_size, world_size)
#         self.skip_bias_add = skip_bias_add
#         self.is_expert = is_expert
#         self.expert_parallel = config.expert_model_parallel_size > 1
#         self.embedding_activation_buffer = embedding_activation_buffer
#         self.grad_output_buffer = grad_output_buffer
#         self.config = config
#         self.shared_expert = shared_expert

#         # Parameters.
#         # Note: torch.nn.functional.linear performs XA^T + b and as a result
#         # we allocate the transpose.
#         # Initialize weight.
#         if not skip_weight_param_allocation:
#             if config.use_cpu_initialization:
#                 self.weight = Parameter(
#                     torch.empty(
#                         self.output_size_per_partition, self.input_size, dtype=config.params_dtype
#                     )
#                 )
#                 if config.perform_initialization:
#                     self.master_weight = _initialize_affine_weight_cpu(
#                         self.weight,
#                         self.output_size,
#                         self.input_size,
#                         self.output_size_per_partition,
#                         0,
#                         init_method,
#                         stride=stride,
#                         return_master_weight=keep_master_weight_for_test,
#                     )
#             else:
#                 self.weight = Parameter(
#                     torch.empty(
#                         self.output_size_per_partition,
#                         self.input_size,
#                         device=torch.cuda.current_device(),
#                         dtype=config.params_dtype,
#                     )
#                 )
#                 if config.perform_initialization:
#                     _initialize_affine_weight_gpu(
#                         self.weight,
#                         init_method,
#                         partition_dim=0,
#                         stride=stride,
#                         expert_parallel=(self.is_expert and self.expert_parallel),
#                     )

#             setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))
#         else:
#             self.weight = None

#         self.register_parameter('bias', None)

#         self.async_tensor_model_parallel_allreduce = (
#             config.async_tensor_model_parallel_allreduce and world_size > 1
#         )

#         self.sequence_parallel = config.sequence_parallel
#         if self.sequence_parallel and world_size <= 1:
#             self.sequence_parallel = False

#         self.gradient_accumulation_fusion = config.gradient_accumulation_fusion

#         if self.async_tensor_model_parallel_allreduce and self.sequence_parallel:
#             raise RuntimeError(
#                 "`async_tensor_model_parallel_allreduce` and `sequence_parallel` "
#                 "cannot be enabled at the same time."
#             )

#         self._forward_impl = linear_with_grad_accumulation_and_async_allreduce
#         self.explicit_expert_comm = self.is_expert and (
#             self.sequence_parallel or self.expert_parallel
#         )

#         # Hook adding a default empty _extra_state for state dict
#         self._register_load_state_dict_pre_hook(
#             lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
#                 f'{prefix}_extra_state'
#             )
#         )

#     def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None):
#         """Forward of ColumnParallelLinear

#         Args:
#             input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

#             weight (optional): weight tensor to use, compulsory when
#                 skip_weight_param_allocation is True.

#         Returns:
#             - output
#             - bias

#         """
#         if weight is None:
#             if self.weight is None:
#                 raise RuntimeError(
#                     "weight was not supplied to ColumnParallelLinear forward pass "
#                     "and skip_weight_param_allocation is True."
#                 )
#             weight = self.weight
#         else:
#             # Check the weight passed in is the correct shape
#             expected_shape = (self.output_size_per_partition, self.input_size)
#             if weight.shape != expected_shape:
#                 raise RuntimeError(
#                     f"supplied weight's shape is {tuple(weight.shape)}, "
#                     f"not {expected_shape} as expected"
#                 )

#         if self.config._cpu_offloading_context is not None:
#             if self.config._cpu_offloading_context.inside_context == True:
#                 assert (
#                     self.config.cpu_offloading == False
#                 ), "CPU Offloading cannot be enabled while using non-TE modules"

#         bias = self.bias if not self.skip_bias_add else None

#         if (
#             self.async_tensor_model_parallel_allreduce
#             or self.sequence_parallel
#             or self.explicit_expert_comm
#         ):
#             input_parallel = input_
#         else:
#             input_parallel = copy_to_tensor_model_parallel_region(input_)

#         if self.config.defer_embedding_wgrad_compute:
#             self.embedding_activation_buffer.append(input_parallel)

#         # Matrix multiply.
#         if not weight.requires_grad:
#             self._forward_impl = linear_with_frozen_weight
#         else:
#             self._forward_impl = linear_with_grad_accumulation_and_async_allreduce

#         output_parallel = self._forward_impl(
#             input=input_parallel,
#             weight=weight,
#             bias=bias,
#             gradient_accumulation_fusion=self.gradient_accumulation_fusion,
#             async_grad_allreduce=False
#             if self.explicit_expert_comm
#             else self.async_tensor_model_parallel_allreduce,
#             sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
#             grad_output_buffer=self.grad_output_buffer
#             if self.config.defer_embedding_wgrad_compute
#             else None,
#             shared_expert=self.shared_expert
#         )
#         if self.gather_output:
#             # All-gather across the partitions.
#             assert not self.sequence_parallel
#             output = gather_from_tensor_model_parallel_region(output_parallel)
#         else:
#             output = output_parallel
#         output_bias = self.bias if self.skip_bias_add else None
#         return output, output_bias

#     def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
#         """ Sharding along axis 0, bias sharded """
#         state_dict = self.state_dict(prefix='', keep_vars=True)
#         return make_sharded_tensors_for_checkpoint(
#             state_dict, prefix, {'weight': 0, 'bias': 0}, sharded_offsets
#         )

#     def set_extra_state(self, state: Any):
#         """ Extra state is ignored """

#     def get_extra_state(self) -> None:
#         """ Keep compatibility with TE state dict. """
#         return None


# class RowParallelLinear(torch.nn.Module):
#     def __init__(
#         self,
#         input_size: int,
#         output_size: int,
#         *,
#         config: ModelParallelConfig,
#         init_method: Callable,
#         bias: bool,
#         input_is_parallel: bool,
#         skip_bias_add: bool,
#         stride: int = 1,
#         keep_master_weight_for_test: bool = False,
#         is_expert: bool = False,
#         tp_comm_buffer_name: str = None,  # Not used
#         shared_expert: bool = False
#     ):
#         super(RowParallelLinear, self).__init__()

#         # Keep input parameters
#         self.input_size = input_size
#         self.output_size = output_size
#         self.input_is_parallel = input_is_parallel
#         # Divide the weight matrix along the last dimension.
#         world_size = get_tensor_model_parallel_world_size()
#         self.input_size_per_partition = divide(input_size, world_size)
#         self.skip_bias_add = skip_bias_add
#         self.config = config
#         self.is_expert = is_expert
#         self.expert_parallel = config.expert_model_parallel_size > 1
#         self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
#         self.sequence_parallel = config.sequence_parallel
#         self.shared_expert = shared_expert
#         if self.sequence_parallel and not self.input_is_parallel:
#             raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

#         # Parameters.
#         # Note: torch.nn.functional.linear performs XA^T + b and as a result
#         # we allocate the transpose.
#         # Initialize weight.
#         if config.use_cpu_initialization:
#             self.weight = Parameter(
#                 torch.empty(
#                     self.output_size, self.input_size_per_partition, dtype=config.params_dtype
#                 )
#             )
#             if config.perform_initialization:
#                 self.master_weight = _initialize_affine_weight_cpu(
#                     self.weight,
#                     self.output_size,
#                     self.input_size,
#                     self.input_size_per_partition,
#                     1,
#                     init_method,
#                     stride=stride,
#                     return_master_weight=keep_master_weight_for_test,
#                     params_dtype=config.params_dtype,
#                 )
#         else:
#             self.weight = Parameter(
#                 torch.empty(
#                     self.output_size,
#                     self.input_size_per_partition,
#                     device=torch.cuda.current_device(),
#                     dtype=config.params_dtype,
#                 )
#             )
#             if config.perform_initialization:
#                 _initialize_affine_weight_gpu(
#                     self.weight,
#                     init_method,
#                     partition_dim=1,
#                     stride=stride,
#                     expert_parallel=(self.is_expert and self.expert_parallel),
#                 )
#         setattr(self.weight, 'allreduce', not (self.is_expert and self.expert_parallel))

#         if bias:
#             if config.use_cpu_initialization:
#                 self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))
#             else:
#                 self.bias = Parameter(
#                     torch.empty(
#                         self.output_size,
#                         device=torch.cuda.current_device(),
#                         dtype=config.params_dtype,
#                     )
#                 )

#             if config.perform_initialization:
#                 # Always initialize bias to zero.
#                 with torch.no_grad():
#                     self.bias.zero_()
#             setattr(self.bias, 'allreduce', not (self.is_expert and self.expert_parallel))
#             setattr(self.bias, 'sequence_parallel', self.sequence_parallel)
#         else:
#             self.register_parameter('bias', None)

#         self._forward_impl = linear_with_grad_accumulation_and_async_allreduce
#         self.explicit_expert_comm = self.is_expert and (
#             self.sequence_parallel or self.expert_parallel
#         )

#         # Hook adding a default empty _extra_state for state dict
#         self._register_load_state_dict_pre_hook(
#             lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
#                 f'{prefix}_extra_state'
#             )
#         )

#     def forward(self, input_):
#         """Forward of RowParallelLinear

#         Args:
#             input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

#         Returns:
#             - output
#             - bias
#         """

#         if self.config._cpu_offloading_context is not None:
#             if self.config._cpu_offloading_context.inside_context == True:
#                 assert (
#                     self.config.cpu_offloading == False
#                 ), "CPU Offloading cannot be enabled while using non-TE modules"

#         # Set up backprop all-reduce.
#         if self.input_is_parallel:
#             input_parallel = input_
#         else:
#             assert not self.sequence_parallel
#             input_parallel = scatter_to_tensor_model_parallel_region(input_)
#         # Matrix multiply.
#         if not self.weight.requires_grad:
#             self._forward_impl = linear_with_frozen_weight
#         else:
#             self._forward_impl = linear_with_grad_accumulation_and_async_allreduce
#         output_parallel = self._forward_impl(
#             input=input_parallel,
#             weight=self.weight,
#             bias=None,
#             gradient_accumulation_fusion=self.gradient_accumulation_fusion,
#             async_grad_allreduce=False,
#             sequence_parallel=False,
#         )

#         # All-reduce across all the partitions.
#         if self.explicit_expert_comm or self.shared_expert:
#             assert self.skip_bias_add
#             output_ = output_parallel
#         elif self.sequence_parallel:
#             output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
#         else:
#             output_ = reduce_from_tensor_model_parallel_region(output_parallel)
#         if not self.skip_bias_add:
#             output = (output_ + self.bias) if self.bias is not None else output_
#             output_bias = None
#         else:
#             output = output_
#             output_bias = self.bias
#         return output, output_bias

#     def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
#         """ Sharding along axis 1, bias not sharded """
#         state_dict = self.state_dict(prefix='', keep_vars=True)
#         return make_sharded_tensors_for_checkpoint(
#             state_dict, prefix, {'weight': 1}, sharded_offsets
#         )

#     def set_extra_state(self, state: Any):
#         """ Extra state is ignored """

#     def get_extra_state(self) -> None:
#         """ Keep compatibility with TE state dict. """
#         return None
