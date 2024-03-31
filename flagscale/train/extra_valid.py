import os
import sys
import math
import torch

from megatron.training.global_vars import get_args
from megatron.training.global_vars import get_tensorboard_writer
from megatron.training.global_vars import get_wandb_writer
from megatron.training.global_vars import get_tokenizer
from megatron.training.utils import print_rank_0 
from megatron.training.utils import print_rank_last 
from megatron.training.utils import is_last_rank
from megatron.core import mpu
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.legacy.data.data_samplers_hetero import build_pretraining_data_loader_hetero


_GLOBAL_EXTRA_VALID_DATASETS = None


def get_extra_valid_datasets():
    """Return extra_valid datasets."""
    return _GLOBAL_EXTRA_VALID_DATASETS


def set_extra_valid_datasets(extra_valid_datasets):
    """Initialize heterogenous context."""
    global _GLOBAL_EXTRA_VALID_DATASETS
    _GLOBAL_EXTRA_VALID_DATASETS = extra_valid_datasets 


def is_dataset_built_on_rank():
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args, data_path):
    tokenizer = get_tokenizer()

    # Only build the validation dataset
    return GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=[data_path],
        blend_per_split=None,
        split="0,1,0",
        path_to_cache=args.data_cache_path,
        mock=args.mock_data,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
    )


def extra_valid_dataset_provider(data_path, num_samples, tag):
    """Build the train test and extra_validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and extra_validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args, data_path)

    if config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0(f"> building extra validation dataset ({data_path}, {num_samples}) for GPT ...")

    extra_train_ds, extra_valid_ds, extra_test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        [0, num_samples, 0],
        config
    ).build()

    assert extra_train_ds is None and extra_test_ds is None, \
        "train_ds and test_ds should be None for extra_valid_ds"

    print_rank_0("> finished creating GPT datasets ...")

    return extra_valid_ds


def build_extra_valid_datasets(build_extra_valid_dataset_provider):
    """Build extra_valid datasets."""

    args = get_args()
    num_tokens_list = args.extra_valid_data_path[0::3]
    paths = args.extra_valid_data_path[1::3]
    tags = args.extra_valid_data_path[2::3]

    num_samples_list = []
    valid_iters_list = []
    for num_tokens in num_tokens_list:
        assert int(num_tokens) > 0, f"Number of tokens {num_tokens} should be greater than 0"
        # Make sure that the number of samples is a multiple of the sequence length
        num_samples = (int(num_tokens) + args.seq_length - 1) // args.seq_length
        # Make sure that the number of samples is a multiple of the global batch size.
        eval_iters = (num_samples + args.global_batch_size - 1) // args.global_batch_size
        num_samples = eval_iters * args.global_batch_size
        num_samples_list.append(num_samples)
        valid_iters_list.append(eval_iters)
    args.extra_valid_iters_list = valid_iters_list
    
    assert len(paths) == len(num_samples_list), \
        f"Number of extra_valid data paths {len(paths)} does not match number of extra_valid data samples {len(num_samples_list)}"
    
    extra_valid_datasets = []
    for path, num_samples, tag in zip(paths, num_samples_list, tags):
        assert os.path.exists(path + ".bin"), f"Path {path} does not exist"
        assert os.path.exists(path + ".idx"), f"Path {path} does not exist"
        extra_valid_datasets.append(build_extra_valid_dataset_provider(path, num_samples, tag))
    
    return extra_valid_datasets


def build_extra_valid_data_loaders(build_extra_valid_dataset_provider):
    """Build extra_valid data loaders."""

    args = get_args()

    paths = args.extra_valid_data_path[1::3]

    extra_valid_dataloaders = [None for _ in paths] 

    print_rank_0('> building extra validation datasets ...')

    # Rely on distributed-aware core datasets, temporary
    is_distributed = getattr(build_extra_valid_dataset_provider, "is_distributed", False)

    # Construct the data pipeline
    if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:

        # Build datasets if necessary.
        if get_extra_valid_datasets() is None:
            extra_valid_datasets = build_extra_valid_datasets(build_extra_valid_dataset_provider)
            set_extra_valid_datasets(extra_valid_datasets)
        else:
            extra_valid_datasets = get_extra_valid_datasets()

        # Build dataloders.
        extra_valid_dataloaders = []
        for extra_valid_ds in extra_valid_datasets:
            if args.hetero_mode != "dp":
                extra_valid_dataloaders.append(build_pretraining_data_loader(extra_valid_ds, 0))
            else:
                extra_valid_dataloaders.append(build_pretraining_data_loader_hetero(extra_valid_ds, 0))

        # Flags to know if we need to do extra_validation.
        do_extra_valid = extra_valid_dataloaders is not None
        flags = torch.tensor([int(do_extra_valid)], dtype=torch.long, device='cuda')
    else:
        flags = torch.tensor([0], dtype=torch.long, device='cuda')

    torch.distributed.broadcast(flags, 0)

    args.do_extra_valid = getattr(args, "do_extra_valid", False) or flags[0].item()

    return extra_valid_dataloaders 


def build_extra_valid_data_iterators(build_extra_valid_dataset_provider):
    """Build pretraining data iterators."""

    # Build loaders.
    extra_valid_dataloaders = \
        build_extra_valid_data_loaders(
            build_extra_valid_dataset_provider)

    if extra_valid_dataloaders[0] is not None:
        extra_valid_data_iterators = []
        for extra_valid_dataloader in extra_valid_dataloaders:
            extra_valid_data_iterators.append(iter(extra_valid_dataloader)) 
    else:
        extra_valid_data_iterators = [None for _ in extra_valid_dataloaders]

    return extra_valid_data_iterators


def extra_evaluate_and_print_results(index, prefix, forward_step_func,
                                     data_iterator, model,
                                     iteration, process_non_loss_data_func, config,
                                     verbose=False, write_to_tensorboard=True):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    if write_to_tensorboard:
        writer = get_tensorboard_writer()
    else:
        writer = None

    wandb_writer = get_wandb_writer()

    # To avoid the circular import.
    from megatron.training import evaluate
    total_loss_dict, collected_non_loss_data, timelimit = evaluate(
        forward_step_func, data_iterator, model,
        process_non_loss_data_func, config, verbose, index)
    

    # Timelimit hit during evaluation
    if timelimit:
        return
    extra_valid_data_path = args.extra_valid_data_path
    path = extra_valid_data_path[1::3][index]
    filename = os.path.basename(path)
    tag = extra_valid_data_path[2::3][index] 
    label = f'{filename}-{tag}'
    string = ' extra_validation {} loss at {} | '.format(label, prefix)
    loss_section = 'validation loss'
    ppl_section = 'validation ppl'
    for key in total_loss_dict:
        string += '{} value: {:.6E} | '.format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += '{} PPL: {:.6E} | '.format(key, ppl)
        if writer:
            writer.add_scalar('{} validation {}'.format(key, label),
                              total_loss_dict[key].item(),
                              iteration)
            writer.add_scalar('{} validation {} vs samples'.format(key, label),
                              total_loss_dict[key].item(),
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'{}/{} validation {}'.format(loss_section, key, label): total_loss_dict[key].item()},
                    iteration)
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar('{} validation {} ppl'.format(key, label), ppl,
                                  iteration)
                writer.add_scalar('{} validation {} ppl vs samples'.format(key, label),
                                  ppl, args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'{}/{} validation {} ppl'.format(ppl_section, key, label): ppl},
                        iteration)

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last('-' * length)
    print_rank_last(string)
    print_rank_last('-' * length)
