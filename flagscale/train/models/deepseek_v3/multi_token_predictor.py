from typing import Literal, Optional

import copy
import torch
from torch import Tensor

from megatron.training import get_args
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import (
    TransformerBlock,
    TransformerBlockSubmodules,
)

from typing import List

from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding

try:
    from megatron.core.extensions.transformer_engine import (
        TENorm,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

def roll_input_mask(input_mask):
    input_mask = torch.roll(input_mask, shifts=-1, dims=0)
    input_mask[-1,:,:] = True
    return input_mask

class DeepSeekSharedEmbedding(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        scatter_to_sequence_parallel: bool = True,
    ):
        super().__init__(config=config)
        
        self.embedding = LanguageModelEmbedding(
            config=self.config,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            position_embedding_type=position_embedding_type,
            scatter_to_sequence_parallel=scatter_to_sequence_parallel,
        )
        self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True

    def forward(
        self,
        input_ids,
        position_ids,
    ) -> Tensor:
        return self.embedding(input_ids=input_ids, position_ids=position_ids)


class DeepSeekSharedHead(MegatronModule):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        config,
        init_method,
        bias=True,
        skip_bias_add=False,
        gather_output=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
    ):
        super().__init__(config=config)
        self.head = tensor_parallel.ColumnParallelLinear(
            hidden_size,
            vocab_size,
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            gather_output=gather_output,
            skip_weight_param_allocation=skip_weight_param_allocation,
            embedding_activation_buffer=embedding_activation_buffer,
            grad_output_buffer=grad_output_buffer,
        )
        self.head.weight.is_embedding_or_output_parameter = True
    
    def forward(
        self,
        hidden_states,
    ) -> Tensor:
        return self.head(hidden_states)


class DeepSeekMultiTokenPredictorLayer(MegatronModule):
    """Multi Token Prediction Layer of DeepSeek V3

    Args:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        scatter_embedding_sequence_parallel: bool = True,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = True,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.embedding = DeepSeekSharedEmbedding(
            config=self.config,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            position_embedding_type=position_embedding_type,
            scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
        )

        if HAVE_TE:   
            self.norm1 = TENorm(config, config.hidden_size, config.layernorm_epsilon)
            self.norm2 = TENorm(config, config.hidden_size, config.layernorm_epsilon)
        else:
            self.norm1 = torch.nn.RMSNorm(normalized_shape=config.hidden_size, eps=config.layernorm_epsilon)
            self.norm2 = torch.nn.RMSNorm(normalized_shape=config.hidden_size, eps=config.layernorm_epsilon)
            
        self.linear_proj = torch.nn.Linear(config.hidden_size*2, config.hidden_size, bias=False)
        
        # the transformer block, fork from main model or use a user-defined transformer layer spec?
        if isinstance(transformer_layer_spec, TransformerBlockSubmodules):
            transformer_layer_spec = transformer_layer_spec.layer_specs[-1]
        self.decoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=True,
        )
        
        self.output_head = DeepSeekSharedHead(
            config.hidden_size,
            vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not parallel_output,
            skip_weight_param_allocation=share_embeddings_and_output_weights,
            embedding_activation_buffer=embedding_activation_buffer,
            grad_output_buffer=grad_output_buffer,
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        pre_hidden_states: Tensor,
        decoder_input: Tensor = None,
        input_mask: Tensor = None,
    ) -> Tensor:
        """Forward pass of the multi token prediction module.

        Args:
            input_ids (Tensor): The input tokens or input embeddings
            pre_hidden_states (Tensor): The hidden states from previous multi token prediction module or main model

        Returns:
            Tensor: The output logits
        """
        ### TODO: fix it
        ### if init self.embedding, but do not use it, will cause gradient sync error in grad bucket of DDP
        assert decoder_input is None, "currently only support embedding input_ids"
        
        if decoder_input is not None:
            pass
        else:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        
        if input_mask is not None:
            # scatter
            input_mask = tensor_parallel.scatter_to_sequence_parallel_region(input_mask)
            decoder_input = torch.masked_fill(decoder_input, input_mask.expand(decoder_input.shape), 0)

        # two RMSNorm
        decoder_input = self.norm1(decoder_input)
        pre_hidden_states = self.norm2(pre_hidden_states)
        # concat
        hidden_states = torch.cat([pre_hidden_states, decoder_input], dim=-1)
        # linear projection
        hidden_states = self.linear_proj(hidden_states)
        # transformer block
        hidden_states = self.decoder(hidden_states, attention_mask)
        hidden_states_mtp = hidden_states
        # output head
        logits_mtp, _ = self.output_head(hidden_states)
        
        return logits_mtp, hidden_states_mtp


class DeepSeekMultiTokenPredictor(MegatronModule):
    """Multi Token Predictor of DeepSeek V3

    Args:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        scatter_embedding_sequence_parallel: bool = True,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = True,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.num_mtp_predictor = config.num_mtp_predictor
        
        self.mtp_modules = torch.nn.ModuleList([
            DeepSeekMultiTokenPredictorLayer(
                    config=self.config,
                    transformer_layer_spec=transformer_layer_spec,
                    vocab_size=vocab_size,
                    max_sequence_length=max_sequence_length,
                    position_embedding_type=position_embedding_type,
                    scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
                    parallel_output=parallel_output,
                    share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                    embedding_activation_buffer=embedding_activation_buffer,
                    grad_output_buffer=grad_output_buffer,
            ) for i in range(self.num_mtp_predictor)
        ])

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        pre_hidden_states: Tensor,
        decoder_input: Tensor = None,
    ) -> Tensor:
        """Forward pass of the multi token prediction module.

        Args:
            input_ids (Tensor): The input tokens or input embeddings
            pre_hidden_states (Tensor): The hidden states from previous multi token prediction module or main model

        Returns:
            Tensor: The output logits
        """
        
        # init input mask
        if decoder_input is not None:
            s, b, _ = decoder_input.shape
        else:
            b, s = input_ids.shape
        input_mask = torch.zeros(s, b).unsqueeze(2).cuda().type(torch.bool)
        input_mask = roll_input_mask(input_mask)
        
        logits_mtps = []
        for i in range(self.num_mtp_predictor):
            logits_mtp, pre_hidden_states = self.mtp_modules[i](
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                pre_hidden_states=pre_hidden_states,
                decoder_input=decoder_input,
                input_mask=input_mask,
            )
            logits_mtps.append(logits_mtp.transpose(0, 1).contiguous())
            input_mask = roll_input_mask(input_mask)
        
        return logits_mtps

