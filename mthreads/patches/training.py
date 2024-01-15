import os
import sys
import json
import time
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

import megatron
from megatron import get_args
from megatron import get_timers
from megatron import print_rank_0
from megatron.initialize import initialize_megatron
from megatron.core.utils import get_model_config
from megatron.checkpointing import save_checkpoint
from megatron.core.enums import ModelType
from megatron.core import mpu, tensor_parallel
from megatron.model import Float16Module
from megatron.model import GPTModel
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.checkpointing import get_checkpoint_name
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint

from megatron.training import print_datetime, setup_model_and_optimizer, build_train_valid_test_data_iterators, \
                              train, evaluate_and_print_results, search_data, mup_prepare

def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
       args.virtual_pipeline_model_parallel_size is not None:
        assert model_type != ModelType.encoder_and_decoder, \
            "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert args.pipeline_model_parallel_split_rank is not None, \
                    "Split rank needs to be specified for model with both encoder and decoder"
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = args.pipeline_model_parallel_split_rank
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (
                        rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process,
                add_encoder=add_encoder,
                add_decoder=add_decoder)
        else:
            model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Disallow training and inference with Transformer Engine
    # for non-GPT models
    args.allow_transformer_engine = all([type(m) == GPTModel for m in model])
    assert args.allow_transformer_engine or args.transformer_impl == 'local', \
        'Transformer Engine is only approved for GPT models'

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    if args.mup == "prepare":
        return model 

    if args.load is not None:
        if args.format_ckpt:
            args.no_load_optim = True
            args.no_load_rng = True
            args.no_save_optim = True
            args.no_save_rng = True

            timers = get_timers()
            timers('format-checkpoint', log_level=0).start(barrier=True)

            args.iteration = load_checkpoint(model, None, None)
            save_checkpoint(args.iteration, model, None, None)

            timers('format-checkpoint').stop(barrier=True)
            timers.log(['format-checkpoint'])
            print_rank_0("Saved checkpoint from other frameworks in Megatron-LM format")
            sys.exit(0)

    # GPU allocation.
    for model_module in model:
        model_module.to('musa:{}'.format(torch.musa.current_device()))

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if wrap_with_ddp:
        if args.DDP_impl == 'torch':
            i = torch.musa.current_device()
            model = [torchDDP(model_module, device_ids=[i], output_device=i,
                              process_group=mpu.get_data_parallel_group())
                     for model_module in model]

        elif args.DDP_impl == 'local':
            model = [LocalDDP(model_module,
                              args.accumulate_allreduce_grads_in_fp32,
                              args.use_contiguous_buffers_in_local_ddp)
                     for model_module in model]
            # broad cast params from data parallel src rank to other data parallel ranks
            if args.data_parallel_random_init:
                for model_module in model:
                    model_module.broadcast_params()

            if args.save_param_index_maps_only:
                if not torch.distributed.is_initialized() \
                   or mpu.get_data_parallel_rank() == 0:
                    # Save param_name_to_index_map
                    param_name_to_index_maps = [] 
                    for model_module in model:
                        param_name_to_index_maps.append(model_module.param_name_to_index_map)
                    # We use iteration 0 to save the param_name_to_index_map
                    index_map_dir = os.path.dirname(get_checkpoint_name(args.save, 0))
                    if not os.path.exists(index_map_dir):
                        os.makedirs(index_map_dir)
                    index_map_file = os.path.join(index_map_dir, "param_name_to_index_maps.json")
                    with open(index_map_file, "w") as f:
                        json.dump(param_name_to_index_maps, f)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                exit(0)
        else:
            raise NotImplementedError('Unknown DDP implementation specified: '
                                      '{}. Exiting.'.format(args.DDP_impl))

    return model

#megatron.training.pretrain = pretrain
megatron.training.get_model = get_model
