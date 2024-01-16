import sys
import time
import torch
import torch_mlu
import megatron
import random
import numpy as np
from megatron import get_args
from datetime import timedelta
from megatron.core import mpu, tensor_parallel
from megatron.arguments import parse_args, validate_args
from megatron.checkpointing import load_args_from_checkpoint
from megatron.global_vars import set_global_variables
from megatron.global_vars import set_hetero_context
from megatron.model.transformer import bias_dropout_add_fused_train
from megatron.model.fused_bias_gelu import bias_gelu
from megatron.utils import save_checkpoint_info

def _compile_dependencies():

    args = get_args()

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling dataset index builder ...")
        from megatron.data.dataset_utils import compile_helper

        compile_helper()
        print(
            ">>> done with dataset index builder. Compilation time: {:.3f} "
            "seconds".format(time.time() - start_time),
            flush=True,
        )

    # ==================
    # Load fused kernels
    # ==================

    # Custom kernel constraints check.
    seq_len = args.seq_length
    attn_batch_size = (
        args.num_attention_heads / args.tensor_model_parallel_size
    ) * args.micro_batch_size
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = (
        seq_len > 16
        and seq_len <= 16384
        and seq_len % 4 == 0
        and attn_batch_size % 4 == 0
    )
    # Print a warning.
    if not (
        (args.fp16 or args.bf16)
        and custom_kernel_constraint
        and args.masked_softmax_fusion
    ):
        if args.rank == 0:
            print(
                "WARNING: constraints for invoking optimized"
                " fused softmax kernel are not met. We default"
                " back to unfused kernel invocations.",
                flush=True,
            )

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling and loading fused kernels ...", flush=True)
        # fused_kernels.load(args)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        # fused_kernels.load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(
            ">>> done with compiling and loading fused kernels. "
            "Compilation time: {:.3f} seconds".format(time.time() - start_time),
            flush=True,
        )


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

def _initialize_distributed():
    """Initialize torch.distributed and core model parallel."""
    args = get_args()

    device_count = torch.mlu.device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert (
                    args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device
            torch.mlu.set_device(device)
    # Call the init process
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(minutes=args.distributed_timeout_minutes),
    )

    if args.hetero_mode is not None:
        # Build the heterogenous context after torch.distributed is initialized and
        # before model parallel is initialized.
        set_hetero_context(args)
        if torch.distributed.get_rank() == 0:
            print(get_hetero_context(), flush=True)

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            if args.hetero_mode is None:
                mpu.initialize_model_parallel(
                    args.tensor_model_parallel_size,
                    args.pipeline_model_parallel_size,
                    args.virtual_pipeline_model_parallel_size,
                    args.pipeline_model_parallel_split_rank,
                    args.fp8 is not None,
                )
            elif args.hetero_mode == "dp":
                mpu.initialize_model_parallel_hetero_dp(
                    args.tensor_model_parallel_size,
                    args.pipeline_model_parallel_size,
                    args.virtual_pipeline_model_parallel_size,
                    args.pipeline_model_parallel_split_rank,
                    args.fp8 is not None,
                )
            elif args.hetero_mode == "pp":
                mpu.initialize_model_parallel_hetero_pp(
                    args.tensor_model_parallel_size,
                    args.pipeline_model_parallel_size,
                    args.virtual_pipeline_model_parallel_size,
                    args.pipeline_model_parallel_split_rank,
                    args.fp8 is not None,
                )
            else:
                raise ValueError(
                    "Hetero mode {} not supported".format(args.hetero_mode)
                )
            if args.rank == 0:
                print(
                    f"> initialized tensor model parallel with size "
                    f"{mpu.get_tensor_model_parallel_world_size()}"
                )
                print(
                    f"> initialized pipeline model parallel with size "
                    f"{mpu.get_pipeline_model_parallel_world_size()}"
                )

def _set_random_seed(seed_, data_parallel_random_init=False):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.mlu.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))

def _warmup_jit_function():
    """Compilie JIT functions before the main training steps"""
    args = get_args()
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Warmup fused bias+gelu
    bias = torch.rand(
        args.ffn_hidden_size // args.tensor_model_parallel_size,
        dtype=dtype,
        device="mlu",
    )
    input = torch.rand(
        (
            args.seq_length,
            args.micro_batch_size,
            args.ffn_hidden_size // args.tensor_model_parallel_size,
        ),
        dtype=dtype,
        device="mlu",
    )
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):
        bias.requires_grad, input.requires_grad = bias_grad, input_grad
        for _ in range(5):
            output = bias_gelu(bias, input)
    del bias, input, output

    # Warmup fused bias+dropout+add
    if args.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length
    input = torch.rand(
        (seq_length, args.micro_batch_size, args.hidden_size),
        dtype=dtype,
        device="mlu",
    )
    residual = torch.rand(
        (seq_length, args.micro_batch_size, args.hidden_size),
        dtype=dtype,
        device="mlu",
    )
    bias = torch.rand((args.hidden_size), dtype=dtype, device="mlu").expand_as(
        residual
    )
    dropout_rate = 0.1
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip(
        [False, True], [True, True], [True, True]
    ):
        input.requires_grad = input_grad
        bias.requires_grad = bias_grad
        residual.requires_grad = residual_grad
        for _ in range(5):
            output = bias_dropout_add_fused_train(input, bias, residual, dropout_rate)
    del bias, input, residual, output
    torch.mlu.empty_cache()

megatron.initialize._compile_dependencies = _compile_dependencies
megatron.initialize._initialize_distributed = _initialize_distributed 
#megatron.initialize._set_random_seed = _set_random_seed 
megatron.initialize._warmup_jit_function= _warmup_jit_function

for k, v in sys.modules.items():
    if 'megatron' in k and hasattr(v, 'set_jit_fusion_options'):
        setattr(v, 'set_jit_fusion_options', set_jit_fusion_options)
