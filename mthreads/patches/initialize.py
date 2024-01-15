import torch
import torch_musa
import random
import os
import numpy as np
import time

import megatron
from datetime import timedelta
from megatron import get_args
from megatron.core import mpu, tensor_parallel

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


def _initialize_distributed():
    """Initialize torch.distributed and core model parallel."""
    args = get_args()

    device_count = torch.musa.device_count()
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
            torch.musa.set_device(device)
    # Call the init process
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(minutes=args.distributed_timeout_minutes),
    )
    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        print("####################################", args.hetero_mode, mpu.model_parallel_is_initialized())
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
            # mpu.initialize_model_parallel(
            #     args.tensor_model_parallel_size,
            #     args.pipeline_model_parallel_size,
            #     args.virtual_pipeline_model_parallel_size,
            #     args.pipeline_model_parallel_split_rank,
            #     args.fp8 is not None,
            # )
            # if args.rank == 0:
            #     print(
            #         f"> initialized tensor model parallel with size "
            #         f"{mpu.get_tensor_model_parallel_world_size()}"
            #     )
            #     print(
            #         f"> initialized pipeline model parallel with size "
            #         f"{mpu.get_pipeline_model_parallel_world_size()}"
            #     )

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
        if torch.musa.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))

def set_jit_fusion_options():
    pass
    
megatron.initialize._compile_dependencies = _compile_dependencies
# megatron.initialize._initialize_distributed = _initialize_distributed
# megatron.initialize._set_random_seed = _set_random_seed

import sys
for k in sys.modules:
    if k.startswith('megatron'):
        if getattr(sys.modules[k], 'set_jit_fusion_options', None):
            setattr(sys.modules[k], 'set_jit_fusion_options', set_jit_fusion_options)
