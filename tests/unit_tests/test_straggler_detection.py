# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import sys
import os
from io import StringIO
from types import SimpleNamespace

import pytest
import torch
import torch.distributed
from flagscale.train.straggler_detection import initialize_straggler_detector

from megatron.core import mpu, parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec as gpt_te_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.dist_checkpointing.models.common import (
    common_test_parallel_reconfiguration_e2e,
)
from tests.unit_tests.test_utilities import Utils


def initialize_gpt_model(
    seed,
    layer_spec_fn=gpt_te_spec,
    vocab_size=128,
    virtual_pipeline_model_parallel_size=None,
    is_moe=False,
    **config_kwargs,
):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    default_config_kwargs = dict(
        num_layers=8,
        hidden_size=128,
        num_attention_heads=8,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    if is_moe:
        transformer_config.moe_layer_freq = [0, 1, 1, 1, 1, 0, 1, 0]
        transformer_config.moe_ffn_hidden_size = 128
        transformer_config.num_moe_experts = 4
        transformer_config.add_bias_linear = False
    model = []
    for i in range(virtual_pipeline_model_parallel_size or 1):
        if is_moe:
            layer_spec = layer_spec_fn(transformer_config, use_transformer_engine=True, vp_stage=i)
        else:
            layer_spec = layer_spec_fn()
        pre_process = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=i)
        post_process = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)
        this_model = (
            GPTModel(
                config=transformer_config,
                transformer_layer_spec=layer_spec,
                vocab_size=vocab_size,
                max_sequence_length=4,
                pre_process=pre_process,
                post_process=post_process,
                position_embedding_type="rope",
                vp_stage=i,
            )
            .bfloat16()
            .cuda()
        )
        this_model.model_type = ModelType.encoder_or_decoder
        model.append(this_model)

    if virtual_pipeline_model_parallel_size is None:
        model = model[0]
    return model


@pytest.fixture
def create_args():
    """Setup dummy args."""
    args = SimpleNamespace()
    args.finetune = False
    args.non_persistent_global_ckpt_dir = None
    args.non_persistent_ckpt_type = None
    args.non_persistent_save_interval = None
    args.exit_on_missing_checkpoint = True
    args.async_save = False
    args.data_parallel_random_init = False
    args.log_progress = False
    args.ckpt_fully_parallel_save = False
    args.ckpt_fully_parallel_load = False
    args.auto_detect_ckpt_format = False
    args.retro_add_retriever = False
    args.ckpt_convert_update_legacy_dist_opt_format = False
    args.ckpt_step = None
    args.use_dist_ckpt = True
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0
    args.vocab_file = None
    args.add_position_embedding = False
    args.ckpt_assume_constant_structure = True
    args.dist_ckpt_strictness = "assume_ok_unexpected"
    args.fp16 = False
    args.bf16 = True
    args.no_save_optim = True
    args.no_save_rng = True
    args.no_load_optim = True
    args.no_load_rng = True
    args.use_distributed_optimizer = True
    args.straggler_detection_level = 2
    args.straggler_detection_warmup_iterations = 50
    args.straggler_detection_interval = 10
    args.curr_iteration = 60

    yield args


# Dense and MoE Models
@pytest.mark.parametrize(
    ('tp_pp_vpp', 'is_moe'),
    [
        ((2, 1, None), False,),
    ],
)
def test_straggler_detection(create_args, tp_pp_vpp, is_moe, monkeypatch):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    args = create_args
    # Model config
    args.num_layers = 8
    args.hidden_size = 128
    args.num_attention_heads = 8
    # Ckpt format
    args.ckpt_format = "torch_dist"
    set_args(args)

    def set_tp_pp_vpp(tp, pp, vpp=None, destroy_first=True):
        if destroy_first:
            Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tp, pp, vpp)
        args.tensor_model_parallel_size = tp
        args.pipeline_model_parallel_size = pp
        args.virtual_pipeline_model_parallel_size = vpp

    set_tp_pp_vpp(*tp_pp_vpp, destroy_first=False)
    init_num_microbatches_calculator(0, None, 1, 1, 1)

    def forward_step_func(data_iterator, model: GPTModel):
        """Forward training step. Copied from `pretrain_gpt.py`"""
        tokens = torch.LongTensor([[2, 1, 2, 3, 4, 5, 7, 6]]).cuda()
        position_ids = torch.arange(8).view(1, -1).cuda()
        attention_mask = None

        output_tensor = model(tokens, position_ids, attention_mask)

        def loss_func(output_tensor: torch.Tensor):
            loss = output_tensor.sum()
            return output_tensor, loss

        return output_tensor, loss_func

    layer_spec_fn = get_gpt_decoder_block_spec if is_moe else gpt_te_spec
    model = initialize_gpt_model(
        1,
        layer_spec_fn=layer_spec_fn,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
        is_moe=is_moe,
    )
    model = model if isinstance(model, list) else [model]

    # Initialize Straggler Detection
    initialize_straggler_detector(user_specified_level=args.straggler_detection_level)

    output_buffer = StringIO()

    if torch.distributed.get_rank() == 0:
        monkeypatch.setattr(sys, 'stdout', output_buffer)

    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[range(0, 100)] * len(model),
        model=model,
        num_microbatches=4,
        seq_length=8,
        micro_batch_size=1,
        forward_only=True,
    )
    
    if torch.distributed.get_rank() == 0:
        captured_output = output_buffer.getvalue()

        print("\n===== Captured Rank0 Report =====")
        print(captured_output)
        print("================================")

        assert captured_output != "", "Rank0 doesn't generate report"
        assert captured_output == "report", "This assert must fail"

    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()
