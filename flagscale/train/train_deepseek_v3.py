# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import sys
from flagscale.utils import CustomModuleFinder
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.meta_path.insert(0, CustomModuleFinder())

import torch
from functools import partial
from contextlib import nullcontext
import inspect

from typing import List, Optional, Tuple, Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel

from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    get_batch_on_this_ulysses_sp_rank,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from flagscale.datasets.sft_dataset import SFTDatasetConfig, SFTDataset
from flagscale.train.extra_valid import extra_valid_datasets_provider
from flagscale.train.train import pretrain
from flagscale.train.global_vars import get_parallel_context

from flagscale.train.models.deepseek_v3.deepseek_v3_model import DeepSeekV3Model
from flagscale.train.models.deepseek_v3.multi_token_predictor import roll_tensor

stimer = StragglerDetector()

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    config = None
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        para_ctx = get_parallel_context()
        if para_ctx is not None:
            config = para_ctx.get_transformer_config()

        if config is None:
            config = core_transformer_config_from_args(args)

    assert args.use_legacy_models is False
    assert args.untie_embeddings_and_output_weights is True, "Current do not support share embeddings and lm head of main model."
    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te)

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            model = DeepSeekV3Model(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling
            )

    return model



def _get_batch_on_this_tp_rank(data_iterator):

    args = get_args()

    def _broadcast(item):
       if item is not None:
           torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

       if data_iterator is not None:
           data = next(data_iterator)
       else:
           data = None

       batch = {
           'tokens': data["tokens"].cuda(non_blocking = True),
           'labels': data["labels"].cuda(non_blocking = True),
           'loss_mask': data["loss_mask"].cuda(non_blocking = True),
           'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking = True),
           'position_ids': data["position_ids"].cuda(non_blocking = True)
       }

       _broadcast(batch['tokens'])
       _broadcast(batch['labels'])
       _broadcast(batch['loss_mask'])
       _broadcast(batch['attention_mask'])
       _broadcast(batch['position_ids'])

    else:

       tokens=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
       labels=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
       loss_mask=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.float32 , device = torch.cuda.current_device())
       if args.create_attention_mask_in_dataloader:
           attention_mask=torch.empty(
                (args.micro_batch_size,1,args.seq_length,args.seq_length), dtype = torch.bool , device = torch.cuda.current_device()
            )
       else:
           attention_mask=None
       position_ids=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())

       _broadcast(tokens)
       _broadcast(labels)
       _broadcast(loss_mask)
       _broadcast(attention_mask)
       _broadcast(position_ids)

       batch = {
           'tokens': tokens,
           'labels': labels,
           'loss_mask': loss_mask,
           'attention_mask': attention_mask,
           'position_ids': position_ids
       }

    return batch


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = _get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    # slice batch along sequence dimension for ulysses sequence parallelism
    batch = get_batch_on_this_ulysses_sp_rank(batch)

    return batch.values()


# define spiky loss as a variation of 20% or more
SPIKY_LOSS_PERC = 0.2



def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()
    loss_mask = loss_mask.view(-1).float() # [b*s]
    total_tokens = loss_mask.sum()

    loss = output_tensor
    use_mtp_predictor = True if args.num_mtp_predictor > 0 else False
    if use_mtp_predictor:
        loss, loss_mtps = output_tensor # [b s]
    roll_loss_mask = loss_mask # [b*s]

    # cal loss for main model
    loss = torch.cat([torch.sum(loss.view(-1) * loss_mask).view(1), total_tokens.view(1)])
    # cal loss for mtp modules
    if use_mtp_predictor:
        loss_mask_mtps = []
        total_tokens_mtps = 0
        for i in range(args.num_mtp_predictor):
            loss_mask_mtp, total_tokens_mtp = roll_tensor(roll_loss_mask, dims=0)
            roll_loss_mask = loss_mask_mtp
            total_tokens_mtps += total_tokens_mtp
            loss_mask_mtps.append(loss_mask_mtp)
        loss_mask_mtps = torch.cat(loss_mask_mtps, 0)
        loss_mtps = torch.cat([torch.sum(loss_mtps.view(-1) * loss_mask_mtps).view(1), total_tokens_mtps.view(1)])
        loss_mtps = loss_mtps / args.num_mtp_predictor

    # merge loss, how to process?
    if use_mtp_predictor:
        loss = loss + loss_mtps

    # loss printing
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(rerun_state_machine.is_spiky_loss, threshold=SPIKY_LOSS_PERC),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
    )


def core_sft_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return SFTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        apply_sft_dataset_separated_loss_mask_if_existed=args.apply_sft_dataset_separated_loss_mask_if_existed,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = None
    para_ctx = get_parallel_context()
    if para_ctx is not None:
        config = para_ctx.get_dataset_config()

    if config is None:
        if args.apply_sft_dataset_separated_loss_mask_if_existed:
            config = core_sft_dataset_config_from_args(args)
        else:
            config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    elif args.apply_sft_dataset_separated_loss_mask_if_existed:
        dataset_type = SFTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    extra_valid_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_valid_dataset_provider=extra_valid_datasets_provider
    )
