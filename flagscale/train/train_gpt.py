# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain and SFT GPT."""

import datetime
import os
import torch

from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
from einops import rearrange

from megatron.core import parallel_state
from megatron.training import get_args
from megatron.training import inprocess_restart
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

import megatron.legacy.model  # isort: skip

# NOTE: Loading `megatron.legacy.model` earlier fails due to circular import

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

from flagscale.train.datasets.sft_dataset import SFTDatasetConfig, SFTDataset
from flagscale.train.extra_valid import extra_valid_datasets_provider
from flagscale.train.train import pretrain
from flagscale.train.global_vars import get_parallel_context

####### magi attention import ######
from magi_attention.api import magi_attn_flex_dispatch, magi_attn_flex_key, undispatch, calc_attn, squash_batch_dim, full_attention_to_varlen_attention, compute_pad_size   # func tools and interface
from magi_attention.api.magi_attn_interface import get_position_ids
from magi_attention.api.functools import pad_at_dim
from magi_attention.common.ranges import AttnRanges
from magi_attention.common.enum import AttnMaskType, AttnOverlapMode, OverlapAlgType
from magi_attention.config import DistAttnConfig
from magi_attention.meta.solver.dispatch_solver import (
    DispatchConfig,
    LBDispatchAlg,
    DPDispatchAlg,
    BSDispatchAlg,
    MinHeapDispatchAlg,
    BTPDispatchAlg,
    ToppHeapDispatchAlg,
)
from magi_attention.meta.solver.overlap_solver import (
    OverlapConfig,
    UniformOverlapAlg,
    GreedyOverlapAlg,
)
MAGI_CHUNK_SIZE = 32
####### magi attention end ######

stimer = StragglerDetector()


def model_provider(
    pre_process=True, post_process=True, vp_stage: Optional[int] = None, is_dualpipev_first_chunk: Optional[bool] = False,
) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return model_provider_modelopt(pre_process, post_process)

    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            dump(
                snapshot,
                open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'),
            )

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

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

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config, use_transformer_engine=use_te, normalization=args.normalization, qk_l2_norm=args.qk_l2_norm, vp_stage=vp_stage, is_dualpipev_first_chunk=is_dualpipev_first_chunk, magi_attention=config.magi_attention,
                )
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts,
                        args.moe_grouped_gemm,
                        args.qk_layernorm,
                        args.multi_latent_attention,
                        args.moe_use_legacy_grouped_gemm,
                        qk_l2_norm=args.qk_l2_norm,
                        use_kitchen=config.use_kitchen,
                        magi_attention=config.magi_attention,
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts,
                        args.moe_grouped_gemm,
                        args.qk_layernorm,
                        args.multi_latent_attention,
                        args.moe_use_legacy_grouped_gemm,
                        normalization=args.normalization,
                        use_kitchen=config.use_kitchen,
                        magi_attention=config.magi_attention,
                    )
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec, use_transformer_engine=use_te, vp_stage=vp_stage
            )

        model = GPTModel(
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
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )

    return model


def new_squash_batch_dim(x):
    assert x.shape[0] == 1, "Magi Attention is not supported with micro_batch_size > 1"
    x_merged = rearrange(x, "b s ... -> (s b) ...")
    return x_merged


def prepare_data(input):
    # input with shape [b, s]
    args = get_args()
    # squash batch dim.
    input = new_squash_batch_dim(input)
    pad_size = compute_pad_size(input.size(0), args.context_parallel_size, head_dim=args.kv_channels, chunk_size=MAGI_CHUNK_SIZE)
    return input, pad_size


def prepare_magi_attention(input, pad_size, cp_group):
    dist_attn_config = DistAttnConfig(
        dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
        overlap_config=OverlapConfig(
            enable=True,
            mode=AttnOverlapMode.STATIC,
            degree=4,
            min_chunk_size=13,
            max_num_chunks=52,
            alg=UniformOverlapAlg(
                random_costs=True,
                random_seed=42,
            ),
        ),
        high_bandwith_domain_size=8,
        deterministic=False,
    )

    args = get_args()
    config = core_transformer_config_from_args(args)

    # fake: text img img text img img text
    # text: 0-10
    # img: 10-100
    # img: 100-300
    # text: 300-400
    # img: 400-600
    # img: 600-1000
    # text: 1000-last
    q_ranges_list = [[0, 10], [10, 100], [100, 400], [100, 400], [400, 600], [400, 600], [600, args.seq_length], [600, args.seq_length], [600, args.seq_length]]
    k_ranges_list = [[0, 10], [0, 100], [0, 10], [100, 400], [0, 10], [100, 600], [0, 10], [100, 400], [600, args.seq_length]]
    attn_mask_type = [AttnMaskType.CAUSAL, AttnMaskType.FULL, AttnMaskType.FULL, AttnMaskType.CAUSAL, AttnMaskType.FULL, AttnMaskType.FULL, AttnMaskType.FULL, AttnMaskType.FULL, AttnMaskType.CAUSAL]
    # print(f"q_ranges_list is {q_ranges_list}")
    # print(f"k_ranges_list is {k_ranges_list}")
    # print(f"attn_mask_type is {attn_mask_type}")

    x_padded, dist_attn_runtime_key = magi_attn_flex_dispatch(
                input,
                q_ranges=AttnRanges.from_ranges(q_ranges_list),
                k_ranges=AttnRanges.from_ranges(k_ranges_list),
                attn_mask_type=attn_mask_type,
                total_seqlen_q=args.seq_length,
                total_seqlen_k=args.seq_length,
                head_dim=args.kv_channels,
                pad_size=pad_size,
                chunk_size=MAGI_CHUNK_SIZE,
                cp_group=cp_group,
                cp_mesh=None,
                dist_attn_config=dist_attn_config,
                is_same_source=True,
                is_q_permutable=True,
                is_k_permutable=True,
          )

    return x_padded, dist_attn_runtime_key


def dispatch_along_cp_rank(batch: Dict[str, Any]):
    """slice data along sequence dimension for context parallelisms and prepare magiattention key."""
    args = get_args()
    tokens = batch['tokens'] #[b, s], b=1
    position_ids = batch['position_ids'] #[b, s], b=1
    assert tokens is not None and position_ids is not None, "tokens and position_ids must be provided for magi attention"

    # tokens
    tokens, pad_size_for_tokens = prepare_data(tokens) #[b*s], [s]
    cp_group = mpu.get_context_parallel_group()
    input, dist_attn_runtime_key = prepare_magi_attention(
        tokens, pad_size_for_tokens, cp_group,
    ) #[s/cp_size]
    # reshape, megatron need batch_dim for input.
    micro_batch_size = args.micro_batch_size
    input = input.view(micro_batch_size, -1) #[b, s], b=1
    batch['tokens'] = input
    
    # position_ids
    if not args.use_rotary_position_embeddings:
        magi_position_ids = get_position_ids(dist_attn_runtime_key)
        orig_position_ids = batch['position_ids']
        batch['position_ids'] = orig_position_ids[magi_position_ids]

    # labels
    # skip processing, logits will be undispatched to be full tensors
    # loss_mask
    # skip processing, logits will be undispatched to be full tensors
    # attention_mask
    # skip processing, will be processed in magi attn initialization

    # others
    batch['key'] = dist_attn_runtime_key

    return batch


def get_batch(data_iterator):
    """Generate a batch."""
    # print(f"in rank {torch.distributed.get_rank()}, call get_batch, data_iterator is {data_iterator}")

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)
    # print(f"get batch, batch is {batch}")

    args = get_args()
    if args.magi_attention:
        batch = dispatch_along_cp_rank(batch)
    else:
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


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
        if args.magi_attention:
            tokens, labels, loss_mask, attention_mask, position_ids, key = get_batch(data_iterator)
            # print(f"in rank {torch.distributed.get_rank()}, after get_batch, tokens is {tokens}, key is {key}")
        else:
            tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if args.magi_attention:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, magi_attention_key=key
                )
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank():
    return parallel_state.get_tensor_model_parallel_rank() == 0


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
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
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
        config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    extra_valid_datasets_provider.is_distributed = True ######## FlagScale ########

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
        extra_valid_dataset_provider=extra_valid_datasets_provider
    )
