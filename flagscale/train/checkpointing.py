import os
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import torch
from megatron.training.global_vars import get_args
from megatron.training.checkpointing import (
    get_rng_state,
    get_checkpoint_name,
    save_checkpoint as megatron_save_checkpoint,
    ensure_directory_exists,
    generate_state_dict,
    get_checkpoint_tracker_filename,
)
from megatron.training.utils import unwrap_model, print_rank_0
from megatron.core.optimizer import DistributedOptimizer
from megatron.core import mpu


def _tensor_to_cpu(tensor):
    """
    Copy Tensor to CPU.
    """
    return tensor.detach().cpu()


def _recursive_to_cpu(data, executor, futures, prefix=""):
    """
    Recursive to copy data to CPU.
    """
    if isinstance(data, torch.Tensor):
        future = executor.submit(_tensor_to_cpu, data)
        futures.append((prefix, future))
    elif isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            _recursive_to_cpu(value, executor, futures, full_key)
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            full_key = f"{prefix}[{idx}]"
            _recursive_to_cpu(value, executor, futures, full_key)
    else:
        futures.append((prefix, data))


def copy_to_cpu(state_dict, filename, max_workers=100):
    results = {}
    futures = []
    start = time.time()
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Copy to cpu for {filename} at {formatted_time}.")
    with ThreadPoolExecutor(max_workers=100) as executor:
        _recursive_to_cpu(state_dict, executor, futures)

        for prefix, future in futures:
            if isinstance(future, torch.Tensor) or not hasattr(future, "result"):
                results[prefix] = future
            else:
                results[prefix] = future.result()
    end = time.time()
    print(f"Copy to cpu for {filename} done in {end-start}s")

    return results


write_lock = threading.Lock()


def write_to_disk_async(state_dict, checkpoint_name, tracker_filename):
    """
    Write state dict to disk async.
    """
    start = time.time()
    current_time = datetime.now()
    current_formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Write to storage for {checkpoint_name} at {current_formatted_time}.")

    torch.save(state_dict, checkpoint_name)
    last_time = datetime.now()
    last_formatted_time = last_time.strftime("%Y-%m-%d %H:%M:%S")
    with write_lock:
        with open(tracker_filename, "a") as f:
            f.write(
                f"Written {checkpoint_name} done from {current_formatted_time} to {last_formatted_time}\n"
            )
    end = time.time()
    print(f"Write to storage for {checkpoint_name} done in {end-start}s")


def get_distributed_optimizer_checkpoint_name(
    checkpoints_path, iteration, release=False
):
    """
    Get the name of the distributed optimizer checkpoint file.
    Reference: https://github.com/intelligent-machine-learning/dlrover/blob/master/dlrover/trainer/torch/flash_checkpoint/megatron_dist_ckpt.py#L298
    """
    if release:
        directory = "release"
    else:
        directory = "iter_{:07d}".format(iteration)

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    common_path = os.path.join(checkpoints_path, directory, f"rank_{rank:05d}")
    return os.path.join(common_path, "distrib_optim.pt")


def save_checkpoint(
    iteration,
    model,
    optimizer,
    opt_param_scheduler,
    num_floating_point_operations_so_far,
    checkpointing_context=None,
    flash=False,
):
    """Save a model checkpoint.

    Checkpointing context is used to persist some checkpointing state
    throughout a single job. Must be initialized externally (not used if None).

    NOTE: The implementation is basically the same as megatron, and reference is
    https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/checkpointing.py#L287.
    If flash parameter is True, FlagScale's built-in flash checkpoint feature will be used refering to dlrover, and reference is
    https://github.com/intelligent-machine-learning/dlrover/blob/master/dlrover/trainer/torch/flash_checkpoint/megatron_dist_ckpt.py#L176
    """
    # If flash is False, use native megatron checkpointing
    if not flash:
        megatron_save_checkpoint(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far,
            checkpointing_context=checkpointing_context,
        )
        return

    args = get_args()
    model = unwrap_model(model)
    ckpt_format = args.dist_ckpt_format if args.use_dist_ckpt else "torch"
    print_rank_0(
        "saving checkpoint at iteration {:7d} to {} in {} format".format(
            iteration, args.save, ckpt_format
        )
    )

    # Collect rng state across data parallel ranks.
    # TODO: @Caozhou1995, use_dist_ckpt is false,
    # but will be consistent with megatron, using args.use_dist_ckpt in the future
    rng_state = get_rng_state()

    # TODO: @Caozhou1995, return_base_dir is false,
    # but will be consistent with megatron, using args.use_dist_ckpt in the future
    checkpoint_name = get_checkpoint_name(args.save, iteration)
    ensure_directory_exists(checkpoint_name)

    # Save distributed optimizer's custom parameter state of every rank.
    optim_checkpoint_name = get_distributed_optimizer_checkpoint_name(
        args.save, iteration
    )
    ensure_directory_exists(optim_checkpoint_name)

    tracker_filename = get_checkpoint_tracker_filename(args.save)
    dist_opt_state = {}
    # Save distributed optimizer's custom parameter state.
    if (
        args.use_distributed_optimizer
        and not args.no_save_optim
        and optimizer is not None
    ):
        if not isinstance(optimizer, DistributedOptimizer):
            raise ValueError(
                f"optimizer should be DistributedOptimizer, but got {type(DistributedOptimizer)}."
            )
        dist_opt_state = optimizer.get_parameter_state_fs_bucket_space()

    # TODO: @Caozhou1995, currently use_dist_ckpt is false,
    # but will be consistent with megatron, using args.use_dist_ckpt and in the future
    # Collect args, model, RNG.
    state_dict = {}

    if (
        not torch.distributed.is_initialized()
        or mpu.get_data_modulo_expert_parallel_rank() == 0
    ):
        state_dict = generate_state_dict(
            args,
            model,
            optimizer,
            opt_param_scheduler,
            rng_state,
            use_dist_ckpt=False,
            iteration=iteration,
            optim_sd_kwargs={},
        )
        state_dict["num_floating_point_operations_so_far"] = (
            num_floating_point_operations_so_far
        )
        state_dict_to_disk = copy_to_cpu(state_dict, checkpoint_name)
        thread = threading.Thread(
            target=write_to_disk_async,
            args=(state_dict_to_disk, checkpoint_name, tracker_filename),
        )
        thread.start()

    if dist_opt_state:
        dist_opt_state_to_disk = copy_to_cpu(dist_opt_state, optim_checkpoint_name)
        thread = threading.Thread(
            target=write_to_disk_async,
            args=(dist_opt_state_to_disk, optim_checkpoint_name, tracker_filename),
        )
        thread.start()

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

