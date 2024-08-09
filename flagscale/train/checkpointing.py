"""
Save and load checkpoint for model training referring to dlrover.
Reference: https://github.com/intelligent-machine-learning/dlrover/blob/master/dlrover/trainer/torch/flash_checkpoint/megatron_dist_ckpt.py
"""

import os
import random
import sys

import torch
import numpy as np
import torch.distributed as dist

from megatron.core import mpu, tensor_parallel
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.training import get_args
from megatron.training.checkpointing import (
    check_checkpoint_args,
    find_checkpoint_rank_0,
    fix_query_key_value_ordering,
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    get_checkpoint_version,
    get_rng_state,
    read_metadata,
    set_checkpoint_version,
    update_num_microbatches,
)
from megatron.training.utils import print_rank_0, unwrap_model

from dlrover.trainer.torch.flash_checkpoint.megatron_dist_ckpt import (
    MegatronDistCheckpointer,
    get_checkpoint_storage,
)
from dlrover.trainer.torch.flash_checkpoint.checkpointer import StorageType
from dlrover.python.common.constants import CheckpointConstant


def save_checkpoint(
    iteration,
    model,
    optimizer,
    opt_param_scheduler,
    num_floating_point_operations_so_far,
    checkpointing_context=None,
    storage_type=StorageType.DISK,
    comm_backend="",
    deletion_strategy=None,
    save_timeout=CheckpointConstant.SAVE_TIMEOUT,
    flash=True,
):
    """
    Save a model checkpoint.
    If flash is True, use flash checkpointing, otherwise use megatron checkpointing.
    """
    if not flash:
        from megatron.training.checkpointing import (
            save_checkpoint as megatron_save_checkpoint,
        )

        megatron_save_checkpoint(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far,
            checkpointing_context,
        )
        return
    args = get_args()
    storage = get_checkpoint_storage(deletion_strategy)
    checkpointer = MegatronDistCheckpointer.singleton_instance(
        args.save,
        storage=storage,
        comm_backend=comm_backend,
        use_distributed_optimizer=args.use_distributed_optimizer,
        save_timeout=save_timeout,
    )

    # Only rank zero of the data parallel writes to the disk.
    model = unwrap_model(model)

    print_rank_0(
        "saving checkpoint at iteration {:7d} to {}".format(iteration, args.save)
    )

    # Collect rng state across data parallel ranks.
    rng_state = get_rng_state()

    # Checkpoint name.
    checkpoint_name = get_checkpoint_name(args.save, iteration)
    optim_checkpoint_name = get_dist_optimizer_checkpoint_name(args.save, iteration)

    model_state_dict = {}
    dist_opter_state = {}

    # Save distributed optimizer's custom parameter state.
    if (
        args.use_distributed_optimizer
        and not args.no_save_optim
        and optimizer is not None
    ):
        if isinstance(optimizer, ChainedOptimizer):
            dist_opter_state = get_chained_optimizer_parameter_state(optimizer)
        else:
            dist_opter_state = get_parameter_state(optimizer)

    # Collect args, model, RNG.
    if (
        not torch.distributed.is_initialized()
        or mpu.get_data_modulo_expert_parallel_rank() == 0
    ):
        # Arguments, iteration, and model.
        model_state_dict["args"] = args
        model_state_dict["checkpoint_version"] = 3.0
        model_state_dict["iteration"] = iteration
        model_state_dict["num_floating_point_operations_so_far"] = (
            num_floating_point_operations_so_far
        )
        if len(model) == 1:
            model_state_dict["model"] = model[0].state_dict_for_save_checkpoint()
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                model_state_dict["model%d" % i] = model[
                    i
                ].state_dict_for_save_checkpoint()

        # Optimizer stuff.
        if not args.no_save_optim:
            if optimizer is not None:
                model_state_dict["optimizer"] = optimizer.state_dict()
            if opt_param_scheduler is not None:
                model_state_dict["opt_param_scheduler"] = (
                    opt_param_scheduler.state_dict()
                )

        # RNG states.
        if not args.no_save_rng:
            model_state_dict["rng_state"] = rng_state

    ckpt_sds = {}
    paths = {}
    if model_state_dict:
        ckpt_sds[CheckpointConstant.MODEL_STATES_NAME] = model_state_dict
        paths[CheckpointConstant.MODEL_STATES_NAME] = checkpoint_name
    if dist_opter_state:
        ckpt_sds[CheckpointConstant.OPTIM_STATES_NAME] = dist_opter_state
        paths[CheckpointConstant.OPTIM_STATES_NAME] = optim_checkpoint_name

    if storage_type == StorageType.MEMORY:
        checkpointer.engine.save_to_memory(iteration, ckpt_sds, paths)
    else:
        checkpointer.engine.save_to_storage(iteration, ckpt_sds, paths)

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def load_checkpoint(
    model,
    optimizer,
    opt_param_scheduler,
    load_arg="load",
    strict=True,
    comm_backend="",
    deletion_strategy=None,
    save_timeout=CheckpointConstant.SAVE_TIMEOUT,
    flash=True,
):
    """
    Load a model checkpoint and return the iteration.
    If flash is True, use flash checkpointing, otherwise use megatron checkpointing.
    """
    if not flash:
        from megatron.training.checkpointing import (
            load_checkpoint as megatron_load_checkpoint,
        )

        megatron_load_checkpoint(
            model,
            optimizer,
            load_arg,
            strict,
        )
        return
    args = get_args()
    load_dir = getattr(args, load_arg)
    storage = get_checkpoint_storage(deletion_strategy)
    checkpointer = MegatronDistCheckpointer.singleton_instance(
        args.save,
        storage=storage,
        comm_backend=comm_backend,
        use_distributed_optimizer=args.use_distributed_optimizer,
        save_timeout=save_timeout,
    )

    model = unwrap_model(model)
    (
        model_state_dict,
        opt_state_dict,
        checkpoint_name,
        release,
    ) = _load_checkpoint_from_memory(checkpointer)

    if not model_state_dict:
        (
            model_state_dict,
            opt_state_dict,
            checkpoint_name,
            release,
        ) = _load_base_checkpoint(load_dir, rank0=False)

    # Checkpoint not loaded.
    if model_state_dict is None:

        # Conditionally exit at this point.
        if args.exit_on_missing_checkpoint:
            print_rank_0(">> '--exit-on-missing-checkpoint' set ... exiting. <<")
            torch.distributed.barrier()
            sys.exit()

        # Iteration and num_floating_point_operations_so_far default to 0.
        return 0, 0

    # Set checkpoint version.
    set_checkpoint_version(model_state_dict.get("checkpoint_version", 0))

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = model_state_dict["iteration"]
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = model_state_dict["total_iters"]
            except KeyError:
                print_rank_0(
                    "A metadata file exists but unable to load "
                    "iteration from checkpoint {}, exiting".format(checkpoint_name)
                )
                sys.exit()
    num_floating_point_operations_so_far = model_state_dict.get(
        "num_floating_point_operations_so_far", 0
    )

    # Check arguments.
    assert args.consumed_train_samples == 0
    assert args.consumed_valid_samples == 0
    if "args" in model_state_dict and not args.finetune:
        checkpoint_args = model_state_dict["args"]
        check_checkpoint_args(checkpoint_args)
        args.consumed_train_samples = getattr(
            checkpoint_args, "consumed_train_samples", 0
        )
        update_num_microbatches(consumed_samples=args.consumed_train_samples)
        args.consumed_valid_samples = getattr(
            checkpoint_args, "consumed_valid_samples", 0
        )
    else:
        print_rank_0("could not find arguments in the checkpoint ...")

    # Model.
    strict = (
        False
        if args.retro_add_retriever or args.transformer_impl == "transformer_engine"
        else strict
    )
    if len(model) == 1:
        model[0].load_state_dict(model_state_dict["model"], strict=strict)
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            model[i].load_state_dict(model_state_dict["model%d" % i], strict=strict)

    # Fix up query/key/value matrix ordering if needed.
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f" checkpoint version {checkpoint_version}")
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            # Load state dict.
            if optimizer is not None:
                optimizer.load_state_dict(model_state_dict["optimizer"])
            if args.use_distributed_optimizer:
                if isinstance(optimizer, ChainedOptimizer):
                    load_chained_optimizer_parameter_state(optimizer, opt_state_dict)
                else:
                    load_parameter_state_from_state_dict(optimizer, opt_state_dict)

            # Load scheduler.
            if opt_param_scheduler is not None:
                if "lr_scheduler" in model_state_dict:  # backward compatbility
                    opt_param_scheduler.load_state_dict(
                        model_state_dict["lr_scheduler"]
                    )
                else:
                    opt_param_scheduler.load_state_dict(
                        model_state_dict["opt_param_scheduler"]
                    )
        except KeyError:
            print_rank_0(
                "Unable to load optimizer from checkpoint {}. "
                "Specify --no-load-optim or --finetune to prevent "
                "attempting to load the optimizer state, "
                "exiting ...".format(checkpoint_name)
            )
            sys.exit()
    else:
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            if "rng_state" in model_state_dict:
                # access rng_state for data parallel rank
                if args.data_parallel_random_init:
                    rng_state = model_state_dict["rng_state"][
                        mpu.get_data_parallel_rank()
                    ]
                else:
                    rng_state = model_state_dict["rng_state"][0]
                random.setstate(rng_state["random_rng_state"])
                np.random.set_state(rng_state["np_rng_state"])
                torch.set_rng_state(rng_state["torch_rng_state"])
                torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
                # Check for empty states array
                if not rng_state["rng_tracker_states"]:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    rng_state["rng_tracker_states"]
                )
            else:  # backward compatability
                random.setstate(model_state_dict["random_rng_state"])
                np.random.set_state(model_state_dict["np_rng_state"])
                torch.set_rng_state(model_state_dict["torch_rng_state"])
                torch.cuda.set_rng_state(model_state_dict["cuda_rng_state"])
                # Check for empty states array
                if not model_state_dict["rng_tracker_states"]:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    model_state_dict["rng_tracker_states"]
                )
        except KeyError:
            print_rank_0(
                "Unable to load rng state from checkpoint {}. "
                "Specify --no-load-rng or --finetune to prevent "
                "attempting to load the rng state, "
                "exiting ...".format(checkpoint_name)
            )
            sys.exit()

    # Some utilities want to load a checkpoint without
    # distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(
        f"  successfully loaded checkpoint from {args.load} "
        f"at iteration {iteration}"
    )

    return iteration, num_floating_point_operations_so_far


def get_dist_optimizer_checkpoint_name(checkpoints_path, iteration, release=False):
    """
    Get distributed optimizer checkpoint name.

    NOTE: This function will be customized by FlagScale, and is the same as dlrover currently.
    """

    if release:
        directory = "release"
    else:
        directory = "iter_{:07d}".format(iteration)

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    common_path = os.path.join(checkpoints_path, directory, f"rank_{rank:05d}")
    return os.path.join(common_path, "distrib_optim.pt")


def get_parameter_state(dist_optimizer):
    """Get parameter state (i.e., parameter & optimizer tensors).

    NOTE: This function will be customized by FlagScale, and is the same as dlrover currently.
    """
    state = {}
    for _, gbuf_range_maps in enumerate(dist_optimizer.gbuf_ranges):

        # Iterate grad buffers (by data type).
        assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
        for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
            for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                state.setdefault(bucket_idx, {})

                # Build contiguous DP rank shards (for param + optim states).
                for model_param, param_range_map in gbuf_range_map["param_map"].items():

                    # Main param & optimizer states.
                    (
                        group_index,
                        group_order,
                    ) = dist_optimizer.model_param_group_index_map[model_param]
                    main_param = dist_optimizer.optimizer.param_groups[group_index][
                        "params"
                    ][group_order]
                    optim_state = dist_optimizer.optimizer.state[main_param]

                    state[bucket_idx].setdefault(group_index, {})
                    state[bucket_idx][group_index].setdefault(group_order, {})

                    tensors = {
                        "param": main_param,
                        **optim_state,
                    }
                    state[bucket_idx][group_index][group_order] = tensors
    return state


def get_chained_optimizer_parameter_state(chained_optimizer):
    """
    NOTE: This function will be customized by FlagScale, and is the same as dlrover currently.
    """
    states = []
    for optimizer in chained_optimizer.chained_optimizers:
        if hasattr(optimizer, "get_parameter_state"):
            state_dict = get_parameter_state(optimizer)
            states.append(state_dict)
        else:
            states.append(None)
    return states


def _load_checkpoint_from_memory(checkpointer):
    """
    NOTE: This function will be customized by FlagScale, and is the same as dlrover currently.
    """
    step, state_dict = checkpointer.engine.load()
    model_state_dict = state_dict.get(CheckpointConstant.MODEL_STATES_NAME, {})
    opt_state_dict = state_dict.get(CheckpointConstant.OPTIM_STATES_NAME, {})
    checkpoint_name = "iter_{:07d}".format(step)
    release = False
    return model_state_dict, opt_state_dict, checkpoint_name, release


def _load_base_checkpoint(load_dir, rank0=False):
    """Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    NOTE: This function will be customized by FlagScale, and is the same as dlrover currently.
    """
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return nothing
    if not os.path.isfile(tracker_filename):
        if not rank0:
            print_rank_0(
                "WARNING: could not find the metadata file {} ".format(tracker_filename)
            )
            print_rank_0(
                "    will not load any checkpoints and will start from " "random"
            )
        return None, None, "", False

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration, release = read_metadata(tracker_filename)

    # Checkpoint.
    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
    else:
        checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
        if release:
            print_rank_0(f" loading release checkpoint from {load_dir}")
        else:
            print_rank_0(
                f" loading checkpoint from {load_dir} at iteration {iteration}"
            )

    dist_opt_checkpoint_name = get_dist_optimizer_checkpoint_name(
        load_dir, iteration, release
    )

    # Load the checkpoint.
    try:
        model_state_dict = torch.load(checkpoint_name, map_location="cpu")
        opt_state_dict = {}
        if os.path.exists(dist_opt_checkpoint_name):
            opt_state_dict = torch.load(dist_opt_checkpoint_name, map_location="cpu")
    except BaseException as e:
        print_rank_0("could not load the checkpoint")
        print_rank_0(e)
        sys.exit()

    return model_state_dict, opt_state_dict, checkpoint_name, release


def load_parameter_state_from_state_dict(dist_optimizer, state_dict):
    """
    Load parameter state (i.e., parameter & optimizer tensors).

    NOTE: This function will be customized by FlagScale, and is the same as dlrover currently.
    """
    # Scatter tensors to all DP ranks.
    for gbuf_idx, gbuf_range_maps in enumerate(dist_optimizer.gbuf_ranges):
        for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
            for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                for model_param, param_range_map in gbuf_range_map["param_map"].items():
                    (
                        group_index,
                        group_order,
                    ) = dist_optimizer.model_param_group_index_map[model_param]
                    main_param = dist_optimizer.optimizer.param_groups[group_index][
                        "params"
                    ][group_order]
                    optim_state = dist_optimizer.optimizer.state[main_param]

                    tensors = {
                        "param": main_param,
                        **optim_state,
                    }

                    restored_tensors = state_dict[bucket_idx][group_index][group_order]
                    for key in tensors.keys():
                        tensors[key].data.copy_(restored_tensors[key])


def load_chained_optimizer_parameter_state(chained_optimizer, states):
    """
    Load the distributed parameter states of all optimizers from a list of states.
    NOTE: This function will be customized by FlagScale, and is the same as dlrover currently.
    """
    for idx, optimizer in enumerate(chained_optimizer.chained_optimizers):
        if not hasattr(optimizer, "load_parameter_state_from_state_dict"):
            continue

        state_dict = states[idx] if states else None
        load_parameter_state_from_state_dict(optimizer, state_dict)
