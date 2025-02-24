from typing import Union

import copy
import torch
from torch import Tensor

from dataclasses import dataclass
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule


@dataclass
class DeepSeekMultiTokenPredictorLayerSubmodules:
    """
    Configuration class for specifying the submodules of a multi token predictor layer.
    """
    norm1: Union[ModuleSpec, type] = IdentityOp
    norm2: Union[ModuleSpec, type] = IdentityOp
    linear_proj: Union[ModuleSpec, type] = IdentityOp
    transformer_layer: Union[ModuleSpec, type] = IdentityOp
    final_norm: Union[ModuleSpec, type] = IdentityOp


class DeepSeekMultiTokenPredictorLayer(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: DeepSeekMultiTokenPredictorLayerSubmodules,
    ):
        super().__init__(config=config)

        self.submodules_config = submodules

        self.norm1 = build_module(
            submodules.norm1,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.norm2 = build_module(
            submodules.norm2,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon
        )
        self.linear_proj = build_module(
            submodules.linear_proj,
            config.hidden_size*2,
            config.hidden_size,
            parallel_mode="duplicated",
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        )
        decoder_config = copy.deepcopy(config)
        decoder_config.pipeline_model_parallel_size = 1
        self.transformer_layer = build_module(
            submodules.transformer_layer,
            config=decoder_config,
            layer_number=1,
            hidden_dropout=None,
        )
        self.final_norm = build_module(
            submodules.norm1,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    def forward(
        self,
        decoder_input: Tensor,
        attention_mask: Tensor,
        pre_hidden_states: Tensor,

    ) -> Tensor:
        """Forward pass of the multi token prediction layer.
        """
        assert decoder_input is not None, "Input ids need to be embedded before mtp predictor"

        decoder_input = self.norm1(decoder_input)
        pre_hidden_states = self.norm2(pre_hidden_states)
        hidden_states = torch.cat([pre_hidden_states, decoder_input], dim=-1)
        hidden_states, _ = self.linear_proj(hidden_states)
        hidden_states, _ = self.transformer_layer(hidden_states, attention_mask=attention_mask)
        hidden_states = self.final_norm(hidden_states)

        return hidden_states


class DeepSeekMultiTokenPredictor(MegatronModule):
    """Multi Token Predictor for DeepSeek V3

    Args:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
    """

    def __init__(
        self,
        config: TransformerConfig,
        deepseek_multi_token_predictor_layer_spec: ModuleSpec,
    ):
        super().__init__(config=config)

        self.config = config
        self.num_mtp_predictor = config.num_mtp_predictor

        self.mtp_modules = torch.nn.ModuleList([
            build_module(
                deepseek_multi_token_predictor_layer_spec,
                config=config,
            ) for i in range(self.num_mtp_predictor)
        ])

    def forward(
        self,
        decoder_input: Tensor,
        attention_mask: Tensor,
        pre_hidden_states: Tensor,
    ) -> Tensor:
        """Forward pass of the multi token prediction module.
        """

        hidden_states_mtps = []
        for i in range(self.num_mtp_predictor):
            decoder_input, _ = roll_tensor(decoder_input, dims=0)
            hidden_states = self.mtp_modules[i](
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                pre_hidden_states=pre_hidden_states,
            )
            hidden_states_mtps.append(hidden_states)
            pre_hidden_states = hidden_states

        return hidden_states_mtps

def roll_tensor(tensor, dims=0):
    rolled_tensor = torch.roll(tensor, shifts=-1, dims=dims)
    index = [slice(None)] * rolled_tensor.ndim
    index[dims] = -1
    index = tuple(index)
    rolled_tensor[index] = 0
    return rolled_tensor, rolled_tensor.sum()
