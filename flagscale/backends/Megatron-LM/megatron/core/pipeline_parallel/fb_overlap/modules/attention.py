# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import torch

from megatron.training import get_args

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
