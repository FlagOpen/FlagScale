# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""A layer that compute logits from hidden_stats."""
from typing import Optional
import torch
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from vllm.model_executor.layers.logits_processor import LogitsProcessor as LogitsProcessorGPU

from omni.adaptors.vllm.distributed.parallel_state import get_local_world_group
from omni.adaptors.vllm.distributed.communication_op import local_rank_all_gather


def _prune_hidden_states(
        hidden_states: torch.Tensor,
        selected_token_indices: torch.Tensor,
) -> torch.Tensor:
    # Adapt: view 2d hidden_states to 1d.
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    if hidden_states.shape[0] == selected_token_indices.shape[0]:
        return hidden_states
    # Adapt end.
    return hidden_states.index_select(0, selected_token_indices)

class LogitsProcessor(LogitsProcessorGPU):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """
    def _get_logits_decode(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        # lm_head does not implement TP, so no need for all_gather and all_to_all
        # adapt: 16DP to 16TP
        hidden_states = local_rank_all_gather(hidden_states, dim=0)
        # adapt end.

        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head,
                                            hidden_states,
                                            bias=embedding_bias)
        # adapt: 16TP to 16DP
        logits =  get_local_world_group().all_to_all(logits)
        # adapt end.

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        return logits
