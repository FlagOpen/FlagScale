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
    x_merged = rearrange(x, "b s ... -> (s b) ...")
    return x_merged

def prepare_data(input):
    # input with shape [b, s]
    args = get_args()
    config = core_transformer_config_from_args(args)

    head_dim = config.hidden_size // config.num_attention_heads
    chunk_size = 32

    # squash batch dim.
    # print(f"train_gpt, prepare_data, before squash, input shape is {input.shape}")
    input = new_squash_batch_dim(input)
    # print(f"train_gpt, prepare_data, after squash, input shape is {input.shape}")
    pad_size = compute_pad_size(input.size(0), args.context_parallel_size, head_dim, chunk_size)
    # print(f"train_gpt, prepare_data, head_dim is {head_dim}, cp size is {args.context_parallel_size}, chunk_size is {chunk_size}")
    # print(f"train_gpt, prepare_data, pad_size is {pad_size}")

    return input, pad_size

def split_attention_mask_naive(
    attention_mask: torch.Tensor
):
    """
    Split a 2-D attention mask into the largest possible rectangular or
    lower-triangular sub-masks, using only PyTorch tensors.

    Parameters
    ----------
    attention_mask : torch.Tensor
        Boolean tensor of shape [seq_length, seq_length].
        True means “token can attend”.

    Returns
    -------
    q_ranges : List[List[int, int]]
        List of (q_start, q_end) in **left-closed, right-open** form.
    k_ranges : List[List[int, int]]
        List of (k_start, k_end) in **left-closed, right-open** form.
    types : List[AttnMaskType]
        Either 'AttnMaskType.FULL' (rectangle) or 'AttnMaskType.CAUSAL' (lower-triangle).
    """
    device = attention_mask.device
    seq_length = attention_mask.size(0)

    used = torch.zeros_like(attention_mask, dtype=torch.bool)

    q_ranges: List[List[int, int]] = []
    k_ranges: List[List[int, int]] = []
    types: List[AttnMaskType] = []

    for q0 in range(seq_length):
        for k0 in range(seq_length):
            print(f"processing location: [{q0}, {k0}]")
            if used[q0, k0] or not attention_mask[q0, k0]:
                continue

            # 1. Try the largest possible rectangle ---------------------------
            q_end = seq_length
            for q1 in range(q0 + 1, seq_length + 1):
                if torch.any(~attention_mask[q0:q1, k0]) or torch.any(used[q0:q1, k0]):
                    q_end = q1
                    break

            k_end = seq_length
            for k1 in range(k0 + 1, seq_length + 1):
                if torch.any(~attention_mask[q0, k0:k1]) or torch.any(used[q0, k0:k1]):
                    k_end = k1
                    break

            if q_end > q0 and k_end > k0:
                sub = attention_mask[q0:q_end, k0:k_end]
                cov = used[q0:q_end, k0:k_end]
                if torch.all(sub) and not torch.any(cov):
                    q_ranges.append((q0, q_end))
                    k_ranges.append((k0, k_end))
                    types.append(AttnMaskType.FULL)
                    used[q0:q_end, k0:k_end] = True
                    continue

            # 2. Rectangle failed; attempt lower-triangle ---------------------
            max_len = min(seq_length - q0, seq_length - k0)
            for l in range(max_len, 0, -1):
                q_end = q0 + l
                k_end = k0 + l
                ref = torch.tril(torch.ones((l, l), dtype=torch.bool, device=device))
                sub = attention_mask[q0:q_end, k0:k_end]
                cov = used[q0:q_end, k0:k_end]
                if torch.equal(sub, ref) and not torch.any(cov):
                    q_ranges.append((q0, q_end))
                    k_ranges.append((k0, k_end))
                    types.append(AttnMaskType.CAUSAL)
                    used[q0:q_end, k0:k_end] = True
                    break

    return q_ranges, k_ranges, types


@torch.no_grad()
def split_attention_mask(
    attention_mask: torch.Tensor,
    max_block_exp: int = 13,          # max block size = 2^max_block_exp
) -> tuple[List[List[int]], List[List[int]], List[AttnMaskType]]:
    """
    Ultra-long-sequence optimized splitter.

    Complexity:  O(seq_length log seq_length) time,
                 O(seq_length)      memory.

    Returns
    -------
    q_ranges : List[List[int]]
        Each element is [q_start, q_end) (left-closed, right-open).
    k_ranges : List[List[int]]
        Each element is [k_start, k_end).
    types    : List[AttnMaskType]
        Mask type of the corresponding block.
    """
    device = attention_mask.device
    seq_len = attention_mask.size(0)
    mask = attention_mask.bool()

    # outputs
    q_ranges: List[List[int]] = []
    k_ranges: List[List[int]] = []
    types: List[AttnMaskType] = []

    # track already-covered positions
    used = torch.zeros(seq_len + 1, seq_len + 1, dtype=torch.bool, device=device)

    # iterate block sizes in decreasing order
    for exp in range(min(max_block_exp, seq_len.bit_length()), 0, -1):
        block = 1 << exp
        # print(f"exp {exp}, block size {block}")
        num_q = (seq_len + block - 1) // block
        num_k = (seq_len + block - 1) // block

        # candidate top-left corners
        q0_all = torch.arange(0, seq_len, block, device=device).view(-1, 1)
        k0_all = torch.arange(0, seq_len, block, device=device).view(1, -1)

        q1_all = torch.minimum(q0_all + block, torch.tensor(seq_len, device=device))
        k1_all = torch.minimum(k0_all + block, torch.tensor(seq_len, device=device))

        # flatten indices
        q0_flat = q0_all.expand(num_q, num_k).reshape(-1)
        k0_flat = k0_all.expand(num_q, num_k).reshape(-1)
        q1_flat = q1_all.expand(num_q, num_k).reshape(-1)
        k1_flat = k1_all.expand(num_q, num_k).reshape(-1)

        # keep valid & uncovered blocks
        valid = (q0_flat < q1_flat) & (k0_flat < k1_flat)
        valid &= ~used[q0_flat, k0_flat]
        if not valid.any():
            continue

        q0, k0, q1, k1 = q0_flat[valid], k0_flat[valid], q1_flat[valid], k1_flat[valid]

        # -------- 1. detect FULL (rectangular) blocks --------
        prefix = mask.cumsum(0).cumsum(1)
        ones = q1 - q0
        zeros = torch.zeros_like(ones)
        area = ones * (k1 - k0)
        ok = prefix[q1 - 1, k1 - 1] \
             - prefix[q0 - 1, k1 - 1] \
             - prefix[q1 - 1, k0 - 1] \
             + prefix[q0 - 1, k0 - 1] == area
        ok &= (q0 == 0) | (k0 == 0)  # boundary guard

        # mark FULL blocks
        for flag, r0, c0, r1, c1 in zip(ok.tolist(), q0.tolist(), k0.tolist(), q1.tolist(), k1.tolist()):
            if flag and not used[r0, c0]:
                q_ranges.append([int(r0), int(r1)])
                k_ranges.append([int(c0), int(c1)])
                types.append(AttnMaskType.FULL)
                used[r0:r1, c0:c1] = True

        # -------- 2. detect CAUSAL (lower-triangular) blocks --------
        tri = torch.tril(torch.ones(block, block, device=device))
        ok_tri = torch.zeros(q0.numel(), dtype=torch.bool, device=device)

        for i in range(q0.numel()):
            r0, c0, r1, c1 = q0[i], k0[i], q1[i], k1[i]
            h = r1 - r0
            w = c1 - c0
            if h != w:
                continue
            sub = mask[r0:r1, c0:c1]
            if torch.equal(sub, tri[:h, :h]) and not used[r0, c0]:
                q_ranges.append([int(r0), int(r1)])
                k_ranges.append([int(c0), int(c1)])
                types.append(AttnMaskType.CAUSAL)
                used[r0:r1, c0:c1] = True

    # deterministic ordering (optional)
    order = [i for i, _ in sorted(enumerate(q_ranges), key=lambda x: (x[1][0], x[1][1]))]
    q_ranges = [q_ranges[i] for i in order]
    k_ranges = [k_ranges[i] for i in order]
    types = [types[i] for i in order]

    return q_ranges, k_ranges, types


def prepare_magi_attention(input, attention_mask, pad_size, cp_group):
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

    head_dim = config.hidden_size // config.num_attention_heads
    total_seq_length = args.micro_batch_size * args.seq_length
    half_seq_length = total_seq_length // 2
    
    # # dispatch input data to each rank and get key.
    # print(f"original attention_mask shape is {attention_mask.shape}")
    attention_mask = ~attention_mask
    q_ranges_list, k_ranges_list, attn_mask_type = split_attention_mask(attention_mask[0][0])
    # print(f"q_ranges_list is {q_ranges_list}")
    # print(f"k_ranges_list is {k_ranges_list}")
    # print(f"attn_mask_type is {attn_mask_type}")


    total_seqlen_q = total_seq_length
    total_seqlen_k = total_seq_length
    chunk_size = 32
    x_padded, dist_attn_runtime_key = magi_attn_flex_dispatch(
                input,
                q_ranges=AttnRanges.from_ranges(q_ranges_list),
                k_ranges=AttnRanges.from_ranges(k_ranges_list),
                attn_mask_type=attn_mask_type,
                total_seqlen_q=total_seqlen_q,
                total_seqlen_k=total_seqlen_k,
                head_dim=head_dim,
                pad_size=pad_size,
                chunk_size=chunk_size,
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
    
    # process tokens
    tokens = batch['tokens']
    attention_mask = batch['attention_mask']
    if tokens is None:
        return

    tokens, pad_size_for_tokens = prepare_data(tokens)
    cp_group = mpu.get_context_parallel_group()
    input, dist_attn_runtime_key = prepare_magi_attention(
                tokens, attention_mask, pad_size_for_tokens, cp_group,
            )
    # print(f"train_gpt, dispatch_along_cp_rank, after prepare_magi_attention, input shape is {input.shape}")
    # reshape, megatron need batch_dim for input.
    args = get_args()
    micro_batch_size = args.micro_batch_size

    input = input.view(args.seq_length // args.context_parallel_size, -1).view(micro_batch_size, -1)
    # print(f"train_gpt, dispatch_along_cp_rank, after view, input shape is {input.shape}")
    batch['tokens'] = input
    
    # process others
    batch['key'] = dist_attn_runtime_key
    batch['position_ids'] = get_position_ids(dist_attn_runtime_key)
    # print(f"dist attn runtime key is {dist_attn_runtime_key}")

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


# def is_dataset_built_on_rank():
#     return (
#         parallel_state.is_pipeline_first_stage(ignore_virtual=True)
#         or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
#     ) and parallel_state.get_tensor_model_parallel_rank() == 0


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
