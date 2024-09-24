# Copyright (c) 2024, BAAI.  All rights reserved.
"""DPO GPT."""

import argparse
from datetime import datetime
import os
import sys
from utils import CustomModuleFinder
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.meta_path.insert(0, CustomModuleFinder())

import torch
from functools import partial

from typing import Union

from megatron.training import (
    get_args,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger)
from megatron.training import print_rank_0
from megatron.training import get_num_microbatches
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.training.theoretical_memory_usage import report_theoretical_memory
from megatron.training.utils import (
    unwrap_model,
    print_rank_last,
    report_memory,
    is_last_rank)
from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from flagscale.datasets.dpo_dataset import DPODatasetConfig, DPODataset
from flagscale.train.extra_valid import extra_valid_dataset_provider

from flagscale.train.train import pretrain, num_floating_point_operations

## Copied from nemo_aligner/utils/distributed.py
@torch.no_grad()
def _compute_distributed_log_softmax(vocab_parallel_logits):
    """Expects a size B x S x V//TP tensor, computes a stable distributed softmax
        return shape B x S x V//TP but softmaxed across the V dimension. More stable than just computing softmax
    """
    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(
        logits_max, op=torch.distributed.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group()
    )

    # Subtract the maximum value.
    vocab_parallel_logits = vocab_parallel_logits - logits_max

    sum_exp_logits = vocab_parallel_logits.exp().sum(-1, keepdim=True).float()

    torch.distributed.all_reduce(
        sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group(),
    )

    return vocab_parallel_logits - sum_exp_logits.log_().to(vocab_parallel_logits.dtype)


## Copied from nemo_aligner/utils/distributed.py
class _DistributedLogprob(torch.autograd.Function):
    """Function to get logprobs out and differentiate through it
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):
        get_vocab_range = tensor_parallel.utils.VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size(-1)
        rank = mpu.get_tensor_model_parallel_rank()
        world_size = mpu.get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0

        # higher stability uses a more numerically stable distributed log_softmax instead of softmax
        log_softmax_output = _compute_distributed_log_softmax(vocab_parallel_logits)
        log_probs = log_softmax_output.clone()
        softmax_output = log_softmax_output.exp_()

        log_probs = torch.gather(log_probs, -1, masked_target.unsqueeze(-1)).squeeze(-1)
        log_probs[target_mask] = 0.0

        torch.distributed.all_reduce(
            log_probs, op=torch.distributed.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group(),
        )

        ctx.save_for_backward(softmax_output, target_mask, masked_target)

        return log_probs

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target_mask, masked_target = ctx.saved_tensors
        partition_vocab_size = softmax.size(-1)

        # 1 if it's the chosen log prob, 0 otherwise
        is_chosen = (~target_mask).unsqueeze(-1) * torch.nn.functional.one_hot(
            masked_target, num_classes=partition_vocab_size
        )

        grad_input = is_chosen.float().sub_(softmax)

        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        # if you add an argument to the forward method, then you must add a corresponding None here
        return grad_input, None


def get_batch_on_this_tp_rank(data_iterator):

    args = get_args()

    def _broadcast(item):
       if item is not None:
           torch.distributed.broadcast(
              item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    def _safe_cat(items, dim=0):
       item1, item2 = items
       if item1 is None or item2 is None:
          return None
       else:
          return torch.cat((item1, item2), dim=dim)

    if mpu.get_tensor_model_parallel_rank() == 0:

       if data_iterator is not None:
           data = next(data_iterator)
       else:
           data = None

       batch = {
           'chosen': data["chosen"].cuda(non_blocking=True),
           'chosen_labels': data["chosen_labels"].cuda(non_blocking=True),
           'chosen_loss_mask': data["chosen_loss_mask"].cuda(non_blocking=True),
           'chosen_attention_mask': data["chosen_attention_mask"].cuda(non_blocking=True),
           'chosen_position_ids': data["chosen_position_ids"].cuda(non_blocking=True),
           'chosen_ref_log_probs': data["chosen_ref_log_probs"].cuda(non_blocking=True),
           'rejected': data["rejected"].cuda(non_blocking=True),
           'rejected_labels': data["rejected_labels"].cuda(non_blocking=True),
           'rejected_loss_mask': data["rejected_loss_mask"].cuda(non_blocking=True),
           'rejected_attention_mask': data["rejected_attention_mask"].cuda(non_blocking=True),
           'rejected_position_ids': data["rejected_position_ids"].cuda(non_blocking=True),
           'rejected_ref_log_probs': data["rejected_ref_log_probs"].cuda(non_blocking=True),
       }

       if args.pipeline_model_parallel_size == 1:
           _broadcast(batch['chosen'])
           _broadcast(batch['chosen_labels'])
           _broadcast(batch['chosen_loss_mask'])
           _broadcast(batch['chosen_attention_mask'])
           _broadcast(batch['chosen_position_ids'])
           _broadcast(batch['chosen_ref_log_probs'])
           _broadcast(batch['rejected'])
           _broadcast(batch['rejected_labels'])
           _broadcast(batch['rejected_loss_mask'])
           _broadcast(batch['rejected_attention_mask'])
           _broadcast(batch['rejected_position_ids'])
           _broadcast(batch['rejected_ref_log_probs'])

       elif mpu.is_pipeline_first_stage():
           _broadcast(batch['chosen'])
           _broadcast(batch['chosen_attention_mask'])
           _broadcast(batch['chosen_position_ids'])
           _broadcast(batch['rejected'])
           _broadcast(batch['rejected_attention_mask'])
           _broadcast(batch['rejected_position_ids'])

       elif mpu.is_pipeline_last_stage():
           _broadcast(batch['chosen_labels'])
           _broadcast(batch['chosen_loss_mask'])
           _broadcast(batch['chosen_attention_mask'])
           _broadcast(batch['chosen_ref_log_probs'])
           _broadcast(batch['rejected_labels'])
           _broadcast(batch['rejected_loss_mask'])
           _broadcast(batch['rejected_attention_mask'])
           _broadcast(batch['rejected_ref_log_probs'])

       chosen = data["chosen"].cuda(non_blocking=True).reshape(
          args.micro_batch_size,args.seq_length)
       chosen_labels = data["chosen_labels"].cuda(non_blocking=True).reshape(
          args.micro_batch_size,args.seq_length)
       chosen_loss_mask = data["chosen_loss_mask"].cuda(non_blocking=True).reshape(
          args.micro_batch_size,args.seq_length)
       chosen_attention_mask = data["chosen_attention_mask"].cuda(non_blocking=True).reshape(
          args.micro_batch_size,1,args.seq_length,args.seq_length)
       chosen_position_ids = data["chosen_position_ids"].cuda(non_blocking=True).reshape(
          args.micro_batch_size,args.seq_length)
       chosen_ref_log_probs = data["chosen_ref_log_probs"].cuda(non_blocking=True).reshape(
          args.micro_batch_size,args.seq_length)

       rejected = data["rejected"].cuda(non_blocking=True).reshape(
          args.micro_batch_size,args.seq_length)
       rejected_labels = data["rejected_labels"].cuda(non_blocking=True).reshape(
          args.micro_batch_size,args.seq_length)
       rejected_loss_mask = data["rejected_loss_mask"].cuda(non_blocking=True).reshape(
          args.micro_batch_size,args.seq_length)
       rejected_attention_mask = data["rejected_attention_mask"].cuda(non_blocking=True).reshape(
          args.micro_batch_size,1,args.seq_length,args.seq_length)
       rejected_position_ids = data["rejected_position_ids"].cuda(non_blocking=True).reshape(
          args.micro_batch_size,args.seq_length)
       rejected_ref_log_probs = data["rejected_ref_log_probs"].cuda(non_blocking=True).reshape(
          args.micro_batch_size,args.seq_length)

       tokens = torch.cat((chosen, rejected), dim=0)
       labels = torch.cat((chosen_labels, rejected_labels), dim=0)
       attention_mask = torch.cat((chosen_attention_mask, rejected_attention_mask), dim=0)
       loss_mask = torch.cat((chosen_loss_mask, rejected_loss_mask), dim=0)
       position_ids = torch.cat((chosen_position_ids, rejected_position_ids), dim=0)
       ref_log_probs = torch.cat((chosen_ref_log_probs, rejected_ref_log_probs), dim=0)

       batch = {
           'tokens': tokens,
           'labels': labels,
           'loss_mask': loss_mask,
           'attention_mask': attention_mask,
           'position_ids': position_ids,
           'ref_log_probs': ref_log_probs
       }

    else:

       chosen = torch.empty((args.micro_batch_size,args.seq_length), dtype=torch.int64 , device=torch.cuda.current_device())
       chosen_labels = torch.empty((args.micro_batch_size,args.seq_length), dtype=torch.int64 , device=torch.cuda.current_device())
       chosen_loss_mask = torch.empty((args.micro_batch_size,args.seq_length), dtype=torch.float32 , device=torch.cuda.current_device())
       chosen_attention_mask = torch.empty((args.micro_batch_size,1,args.seq_length,args.seq_length), dtype=torch.bool , device=torch.cuda.current_device())
       chosen_position_ids = torch.empty((args.micro_batch_size,args.seq_length), dtype=torch.int64 , device=torch.cuda.current_device())
       chosen_ref_log_probs = torch.empty((args.micro_batch_size,args.seq_length), dtype=torch.float32 , device=torch.cuda.current_device())

       rejected = torch.empty((args.micro_batch_size,args.seq_length), dtype=torch.int64 , device=torch.cuda.current_device())
       rejected_labels = torch.empty((args.micro_batch_size,args.seq_length), dtype=torch.int64 , device=torch.cuda.current_device())
       rejected_loss_mask = torch.empty((args.micro_batch_size,args.seq_length), dtype=torch.float32 , device=torch.cuda.current_device())
       rejected_attention_mask = torch.empty((args.micro_batch_size,1,args.seq_length,args.seq_length), dtype=torch.bool , device=torch.cuda.current_device())
       rejected_position_ids = torch.empty((args.micro_batch_size,args.seq_length), dtype=torch.int64 , device=torch.cuda.current_device())
       rejected_ref_log_probs = torch.empty((args.micro_batch_size,args.seq_length), dtype=torch.float32 , device=torch.cuda.current_device())

       if args.pipeline_model_parallel_size == 1:
           _broadcast(chosen)
           _broadcast(chosen_labels)
           _broadcast(chosen_loss_mask)
           _broadcast(chosen_attention_mask)
           _broadcast(chosen_position_ids)
           _broadcast(chosen_ref_log_probs)
           _broadcast(rejected)
           _broadcast(rejected_labels)
           _broadcast(rejected_loss_mask)
           _broadcast(rejected_attention_mask)
           _broadcast(rejected_position_ids)
           _broadcast(rejected_ref_log_probs)
 
       elif mpu.is_pipeline_first_stage():
           chosen_labels=None
           chosen_loss_mask=None
           chosen_ref_log_probs=None
           rejected_labels=None
           rejected_loss_mask=None
           rejected_ref_log_probs=None
   
           _broadcast(chosen_tokens)
           _broadcast(chosen_attention_mask)
           _broadcast(chosen_position_ids)
           _broadcast(rejected_tokens)
           _broadcast(rejected_attention_mask)
           _broadcast(rejected_position_ids)

       elif mpu.is_pipeline_last_stage():
           chosen=None
           chosen_position_ids=None
           rejected=None
           rejected_position_ids=None

           _broadcast(chosen_labels)
           _broadcast(chosen_loss_mask)
           _broadcast(chosen_attention_mask)
           _broadcast(chosen_ref_log_probs)
           _broadcast(rejected_labels)
           _broadcast(rejected_loss_mask)
           _broadcast(rejected_attention_mask)
           _broadcast(rejected_ref_log_probs)
 
       tokens = _safe_cat((chosen, rejected), dim=0)
       labels = _safe_cat((chosen_labels, rejected_labels), dim=0)
       attention_mask = _safe_cat((chosen_attention_mask, rejected_attention_mask), dim=0)
       loss_mask = _safe_cat((chosen_loss_mask, rejected_loss_mask), dim=0)
       position_ids = _safe_cat((chosen_position_ids, rejected_position_ids), dim=0)
       ref_log_probs = _safe_cat((chosen_ref_log_probs, rejected_ref_log_probs), dim=0)

       batch = {
           'tokens': tokens,
           'labels': labels,
           'loss_mask': loss_mask,
           'attention_mask': attention_mask,
           'position_ids': position_ids,
           'ref_log_probs': ref_log_probs
       }

    return batch


def train_step(forward_step_func, data_iterator,
               model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size*2,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False)

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    # Vision momentum.
    if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        if args.hetero_mode != "dp":
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
        else:
            micro_batch_for_all_data_parallel = sum(map(lambda x, y: x * y, 
                                                        args.hetero_micro_batch_sizes,
                                                        args.hetero_data_parallel_splits))
            increment = get_num_microbatches() * \
                        micro_batch_for_all_data_parallel
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0].keys():
            numerator = 0
            denominator = 0
            for x in losses_reduced:
                val = x[key]
                # there is one dict per microbatch. in new reporting, we average
                # over the total number of tokens across the global batch.
                if isinstance(val, tuple) or isinstance(val, list):
                    numerator += val[0]
                    denominator += val[1]
                else:
                    # legacy behavior. we average over the number of microbatches,
                    # and so the denominator is 1.
                    numerator += val
                    denominator += 1
            loss_reduced[key] = numerator / denominator
        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    return {}, skipped_iter, grad_norm, num_zeros_in_grad

def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    one_logger = get_one_logger()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'all-grads-sync',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    if args.hetero_mode != "dp":
        batch_size = args.micro_batch_size * args.data_parallel_size * \
            get_num_microbatches()
    else:
        micro_batch_for_all_data_parallel = sum(map(lambda x, y: x * y, 
                                                    args.hetero_micro_batch_sizes,
                                                    args.hetero_data_parallel_splits))
        batch_size = micro_batch_for_all_data_parallel * get_num_microbatches()

    # Track app tag & app tag ID
    if one_logger:
        job_name = os.environ.get('SLURM_JOB_NAME', None)
        current_app_tag = f'{job_name}_{batch_size}_{args.world_size}'
        one_logger.log_app_tag(current_app_tag)

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if is_last_rank() and (iteration % args.tensorboard_log_interval == 0):
        if wandb_writer:
            wandb_writer.log({'samples vs steps': args.consumed_train_samples},
                             iteration)
            wandb_writer.log({'consumed-tokens': args.consumed_train_samples * args.seq_length / 1000. / 1000 / 1000}, iteration)
        if args.log_learning_rate_to_tensorboard:
            if writer:
                writer.add_scalar('learning-rate', learning_rate, iteration)
                if args.decoupled_lr is not None:
                    writer.add_scalar('decoupled-learning-rate', decoupled_learning_rate, iteration)
                writer.add_scalar('learning-rate vs samples', learning_rate,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'learning-rate': learning_rate}, iteration)
        if args.log_batch_size_to_tensorboard:
            if writer:
                writer.add_scalar('batch-size', batch_size, iteration)
                writer.add_scalar('batch-size vs samples', batch_size,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'batch-size': batch_size}, iteration)
        for key in loss_dict:
            if writer:
                writer.add_scalar(key , loss_dict[key], iteration)
                writer.add_scalar(key + ' vs samples', loss_dict[key],
                                  args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({key: loss_dict[key]}, iteration)
        if args.log_loss_scale_to_tensorboard:
            if writer:
                writer.add_scalar('loss-scale', loss_scale, iteration)
                writer.add_scalar('loss-scale vs samples', loss_scale,
                                  args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'loss-scale': loss_scale}, iteration)
        if args.log_world_size_to_tensorboard:
            if writer:
                writer.add_scalar('world-size', args.world_size, iteration)
                writer.add_scalar('world-size vs samples', args.world_size,
                                  args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'world-size': args.world_size}, iteration)
        if grad_norm is not None:
            if writer:
                writer.add_scalar('grad-norm', grad_norm, iteration)
                writer.add_scalar('grad-norm vs samples', grad_norm,
                                  args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'grad-norm': grad_norm}, iteration)
        if num_zeros_in_grad is not None:
            if writer:
                writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
                writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                                  args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'num-zeros': num_zeros_in_grad}, iteration)
        if params_norm is not None:
            if writer:
                writer.add_scalar('params-norm', params_norm, iteration)
                writer.add_scalar('params-norm vs samples', params_norm,
                                  args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'params-norm': params_norm}, iteration)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            if writer:
                writer.add_scalar(
                    "mem-reserved-bytes",
                    mem_stats["reserved_bytes.all.current"],
                    iteration,
                )
                writer.add_scalar(
                    "mem-allocated-bytes",
                    mem_stats["allocated_bytes.all.current"],
                    iteration,
                )
                writer.add_scalar(
                    "mem-allocated-count",
                    mem_stats["allocation.all.current"],
                    iteration,
                )
            if wandb_writer:
                wandb_writer.log(
                    {"mem-reserved-bytes": mem_stats["reserved_bytes.all.current"]},
                    iteration,
                )
                wandb_writer.log(
                    {"mem-allocated-bytes": mem_stats["allocated_bytes.all.current"]},
                    iteration,
                )
                wandb_writer.log(
                    {"mem-allocated-count": mem_stats["allocation.all.current"]},
                    iteration,
                )

    if args.num_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        track_moe_metrics(moe_loss_scale, iteration, writer, wandb_writer, total_loss_dict, args.moe_per_layer_logging)

    if iteration % args.log_interval == 0:
        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        throughput = num_floating_point_operations(args, batch_size) / (
            elapsed_time_per_iteration * 10**12 * args.world_size)
        if args.log_timers_to_tensorboard:
            if writer:
                writer.add_scalar('iteration-time',
                                  elapsed_time_per_iteration, iteration)
            if wandb_writer:
                wandb_writer.log({'iteration-time': elapsed_time_per_iteration},
                                 iteration)
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        if args.log_throughput:
            log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar('throughput', throughput, iteration)
                if wandb_writer:
                    wandb_writer.log({'throughput': throughput}, iteration)
        assert learning_rate is not None
        # Decoupled_learning_rate should be not None only on first and last pipeline stage.
        log_string += ' learning rate: {:.6E} |'.format(learning_rate)
        if args.decoupled_lr is not None and (mpu.is_pipeline_first_stage(ignore_virtual=True) or
                                              mpu.is_pipeline_last_stage(ignore_virtual=True)):
            assert decoupled_learning_rate is not None
            log_string += ' decoupled learning rate: {:.6E} |'.format(decoupled_learning_rate)
        else:
            assert decoupled_learning_rate is None
        log_string += ' global batch size: {:5d} |'.format(batch_size)
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
        log_string += ' loss scale: {:.1f} |'.format(loss_scale)
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)
        if num_zeros_in_grad is not None:
            log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
        if params_norm is not None:
            log_string += ' params norm: {:.3f} |'.format(params_norm)
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if not args.auto_tune:
            if report_memory_flag and learning_rate > 0.0:
                # Report memory after optimizer state has been initialized.
                if torch.distributed.get_rank() == 0:
                    num_microbatches = get_num_microbatches()
                    report_theoretical_memory(
                        args, num_microbatches=num_microbatches, verbose=True
                    )
                report_memory("(after {} iterations)".format(iteration))
                report_memory_flag = False
        else:
            report_memory("(after {} iterations)".format(iteration))
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


stimer = StragglerDetector()

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

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
        )
    else:
        assert (
            args.context_parallel_size == 1
        ), "Context parallelism is only supported with Megatron Core!"

        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def loss_func(loss_mask: torch.Tensor, labels: torch.Tensor, ref_log_probs: torch.Tensor, output_tensor: torch.Tensor):
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

    pi_log_probs = _DistributedLogprob.apply(output_tensor, labels).contiguous()

    log_probs = pi_log_probs - ref_log_probs
    rewards = (log_probs * loss_mask).sum(-1) * args.ref_policy_kl_penalty

    chosen_rewards, reject_rewards = torch.split(rewards.float(), len(output_tensor) // 2, dim=0)

    losses = -torch.nn.functional.logsigmoid(chosen_rewards - reject_rewards).mean(0)
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()

    with torch.no_grad():
        comp = chosen_rewards > reject_rewards
        acc_chosen = comp.float().mean()
        chosen_rewards = chosen_rewards.float().mean()
        reject_rewards = reject_rewards.float().mean()
    
    loss = torch.cat([
        torch.sum(losses.view(-1)).view(1),
        total_tokens.view(1), 
        acc_chosen.view(1),
        chosen_rewards.view(1),
        reject_rewards.view(1),
    ])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {
            #'lm loss': (reporting_loss[0], reporting_loss[1]),
            'lm loss': (reporting_loss[0], 1),
            'chosen acc': (reporting_loss[2], 1),
            'chosen rewards': (reporting_loss[3], 1),
            'reject rewards': (reporting_loss[4], 1),
         },
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
        tokens, labels, loss_mask, attention_mask, position_ids, ref_log_probs = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    # Get logits
    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=None)

    return output_tensor, partial(loss_func, loss_mask, labels, ref_log_probs)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_dpo_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return DPODatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_dpo_dataset_config_from_args(args)

    dataset_type = DPODataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_dpo_args(parser):
    group = parser.add_argument_group(title='dpo')

    group.add_argument('--ref-policy-kl-penalty', type=float, default=0.6,
                       help='KL penalty coeff of ref policy in DPO training.')

    return parser


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    extra_valid_dataset_provider.is_distributed = True

    from flagscale.train import train
    train.train_step = train_step
    train.training_log = training_log

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
             get_batch_fn=get_batch,
             extra_valid_dataset_provider=extra_valid_dataset_provider,
             extra_args_provider=get_dpo_args)

