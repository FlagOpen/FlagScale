# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import torch

from megatron.training import get_args
from megatron.core.pipeline_parallel.fb_overlap.modules.utils import async_all_to_all

AsyncAll2All_INPUT = []
AsyncAll2All_OUTPUT = []


def set_async_alltoall_inputs(*args):
    AsyncAll2All_INPUT.append(args)


def get_async_alltoall_outputs():
    return AsyncAll2All_OUTPUT.pop(0)


def launch_async_all2all():
    global AsyncAll2All_INPUT
    global AsyncAll2All_OUTPUT
    if len(AsyncAll2All_INPUT) > 0:
        input_, input_splits, output_splits, group = AsyncAll2All_INPUT.pop(0)
        _, output, a2a_handle = async_all_to_all(
            input_,
            input_splits,
            output_splits,
            group
        )
        AsyncAll2All_OUTPUT.append((output, a2a_handle))


def launch_async_all2all_hook(_):
    launch_async_all2all()


def attention_forward(
        self,
        hidden_states,
        residual,
        attention_mask=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        recompute_norm=False
):
    # Optional Input Layer norm
    def pre_norm(hidden_states):
        args = get_args()
        input_layernorm_output = self.input_layernorm(hidden_states)
        if getattr(args, 'input_layernorm_in_fp32', False):
            input_layernorm_output = input_layernorm_output.float()
        return input_layernorm_output

    # # skip recompute
    if recompute_norm:
        assert not recompute_norm, "not support recompute norm"
    else:
        input_layernorm_output = pre_norm(hidden_states)

    # Self attention.
    attention_output_with_bias = self.self_attention(
        input_layernorm_output,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
    )

    # TODO: could we move `bias_dropout_add_exec_handler` itself
    # inside the module provided in the `bias_dropout_add_spec` module?
    with self.bias_dropout_add_exec_handler():
        hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
            attention_output_with_bias, residual, self.hidden_dropout
        )

    if recompute_norm:
        self.norm_ckpt1.discard_output()
        hidden_states.register_hook(self.norm_ckpt1.recompute)

    return hidden_states
