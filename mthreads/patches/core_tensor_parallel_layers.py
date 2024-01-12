import warnings
from typing import Callable, Optional

import torch
from torch.nn.parameter import Parameter
from torch.cuda.amp import custom_bwd, custom_fwd

import megatron
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.layers import (
    VocabParallelEmbedding,
    ColumnParallelLinear,
    RowParallelLinear,
    _initialize_affine_weight_gpu,
    _initialize_affine_weight_cpu,
    set_tensor_model_parallel_attributes,
    linear_with_grad_accumulation_and_async_allreduce
)
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.layers.utils import VocabUtility, divide
_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

def VocabParallelEmbedding_init(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: Callable,
        config: ModelParallelConfig,
    ):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings, get_tensor_model_parallel_rank(), self.tensor_model_parallel_size
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        # Allocate weights and initialize.
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition, self.embedding_dim, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.num_embeddings,
                    self.embedding_dim,
                    self.num_embeddings_per_partition,
                    0,
                    init_method,
                    params_dtype=config.params_dtype,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device='musa:{}'.format(torch.musa.current_device()),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)

@staticmethod
@custom_bwd                
def LinearWithGradAccumulationAndAsyncCommunication_backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    use_bias = ctx.use_bias

    if ctx.sequence_parallel:
        world_size = get_tensor_model_parallel_world_size()
        dim_size = list(input.size())
        dim_size[0] = dim_size[0] * world_size

        all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
        handle = torch.distributed._all_gather_base(
            all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
        )

        # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
        # gather is scheduled before the input gradient computation
        total_input = all_gather_buffer
    else:
        total_input = input
    grad_input = grad_output.matmul(weight)

    if ctx.sequence_parallel:
        handle.wait()

    # Doing gather + slicing during the NeMo forward pass can make this tensor
    # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
    # clones it if it's not contiguous:
    # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
    grad_output = grad_output.contiguous()
    # Convert the tensor shapes to 2D for execution compatibility
    grad_output = grad_output.view(
        grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
    )
    total_input = total_input.view(
        total_input.shape[0] * total_input.shape[1], total_input.shape[2]
    )

    if ctx.async_grad_allreduce:
        # Asynchronous all-reduce
        handle = torch.distributed.all_reduce(
            grad_input, group=get_tensor_model_parallel_group(), async_op=True
        )
        # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
        # all-reduce is scheduled before the weight gradient computation

    if ctx.sequence_parallel:
        assert not ctx.async_grad_allreduce
        dim_size = list(input.size())
        sub_grad_input = torch.empty(
            dim_size, dtype=input.dtype, device='musa:{}'.format(torch.musa.current_device()), requires_grad=False
        )
        # reduce_scatter
        handle = torch.distributed._reduce_scatter_base(
            sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
        )
        # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
        # reduce scatter is scheduled before the weight gradient computation

    if ctx.gradient_accumulation_fusion:
        if weight.main_grad.dtype == torch.float32:
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                total_input, grad_output, weight.main_grad
            )
        elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                total_input, grad_output, weight.main_grad
            )
        else:
            raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
        grad_weight = None
    else:
        grad_weight = grad_output.t().matmul(total_input)
    grad_bias = grad_output.sum(dim=0) if use_bias else None

    if ctx.sequence_parallel:
        handle.wait()
        return sub_grad_input, grad_weight, grad_bias, None, None, None

    if ctx.async_grad_allreduce:
        handle.wait()

    return grad_input, grad_weight, grad_bias, None, None, None

def ColumnParallelLinear_init(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
    ):
    super(ColumnParallelLinear, self).__init__()

    # Keep input parameters
    self.input_size = input_size
    self.output_size = output_size
    self.gather_output = gather_output
    # Divide the weight matrix along the last dimension.
    world_size = get_tensor_model_parallel_world_size()
    self.output_size_per_partition = divide(output_size, world_size)
    self.skip_bias_add = skip_bias_add
    self.config = config

    # Parameters.
    # Note: torch.nn.functional.linear performs XA^T + b and as a result
    # we allocate the transpose.
    # Initialize weight.
    if not skip_weight_param_allocation:
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition, self.input_size, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.output_size_per_partition,
                    0,
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    device='musa:{}'.format(torch.musa.current_device()),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight, init_method, partition_dim=0, stride=stride
                )
    else:
        self.weight = None

    if bias:
        if config.use_cpu_initialization:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
            )
        else:
            self.bias = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    device='musa:{}'.format(torch.musa.current_device()),
                    dtype=config.params_dtype,
                )
            )
        set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
        if config.perform_initialization:
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
    else:
        self.register_parameter('bias', None)

    self.async_tensor_model_parallel_allreduce = (
        config.async_tensor_model_parallel_allreduce and world_size > 1
    )

    self.sequence_parallel = config.sequence_parallel
    if self.sequence_parallel and world_size <= 1:
        warnings.warn(
            f"`sequence_parallel` is set to `True`, but tensor model parallel size is {world_size}. "
            f"Disabling sequence parallel."
        )
        self.sequence_parallel = False

    if config.gradient_accumulation_fusion and not _grad_accum_fusion_available:
        raise RuntimeError(
            "ColumnParallelLinear was called with gradient_accumulation_fusion set "
            "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
            "module is not found. To use gradient_accumulation_fusion you must "
            "install APEX with --cpp_ext and --cuda_ext. For example: "
            "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\" "
            "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
            "gradient accumulation fusion."
        )
    self.gradient_accumulation_fusion = config.gradient_accumulation_fusion

    if self.async_tensor_model_parallel_allreduce and self.sequence_parallel:
        raise RuntimeError(
            "`async_tensor_model_parallel_allreduce` and `sequence_parallel` "
            "cannot be enabled at the same time."
        )

    self._forward_impl = linear_with_grad_accumulation_and_async_allreduce


def RowParallelLinear_init(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool = True,
        input_is_parallel: bool = False,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        skip_bias_add: bool = False,
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.config = config
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size, self.input_size_per_partition, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.input_size_per_partition,
                    1,
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                    params_dtype=config.params_dtype,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device='musa:{}'.format(torch.musa.current_device()),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight, init_method, partition_dim=1, stride=stride
                )
        if bias:
            if config.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device='musa:{}'.format(torch.musa.current_device()),
                        dtype=config.params_dtype,
                    )
                )
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)

            if config.perform_initialization:
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        self._forward_impl = linear_with_grad_accumulation_and_async_allreduce


megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__ = VocabParallelEmbedding_init
megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward = LinearWithGradAccumulationAndAsyncCommunication_backward
megatron.core.tensor_parallel.layers.ColumnParallelLinear.__init__ = ColumnParallelLinear_init
megatron.core.tensor_parallel.layers.RowParallelLinear.__init__ = RowParallelLinear_init