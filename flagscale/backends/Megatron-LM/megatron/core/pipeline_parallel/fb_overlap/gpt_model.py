# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import logging
from typing import Dict, Literal, Optional, Tuple, Union, List

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import WrappedTensor, deprecate_inference_params
from .transformer_block import (
    transformer_block_backward, transformer_block_forward_backward_overlaping,
    transformer_block_forward
)
from .modules.utils import (
    LayerGraph, ModelGraph, detach_tensor, run_graph_backward
)


def gpt_model_forward(
    self,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_context: BaseInferenceContext = None,
    inference_params: InferenceParams = None,
    packed_seq_params: PackedSeqParams = None,
    extra_block_kwargs: dict = None,
) -> Tensor:
    # print(f"in gpt_model_forward, start")

    """Forward function of the GPT Model This function passes the input tensors
    through the embedding layer, and then the decoeder and finally into the post
    processing layer (optional).

    It either returns the Loss values if labels are given  or the final hidden units
    """
    # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
    # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

    inference_context = deprecate_inference_params(inference_context, inference_params)
                                                   
    # Decoder embedding.
    if decoder_input is not None:
        preprocess_graph = None
    elif self.pre_process:
        decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        preprocess_graph = decoder_input
    else:
        # intermediate stage of pipeline
        # decoder will get hidden_states from encoder.input_tensor
        decoder_input = None
        preprocess_graph = None

    # Rotary positional embeddings (embedding is None for PP intermediate devices)
    rotary_pos_emb = None
    rotary_pos_cos = None
    rotary_pos_sin = None
    if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
        if not self.training and self.config.flash_decode and inference_context:
            assert (
                inference_context.is_static_batching()
            ), "GPTModel currently only supports static inference batching."
            # Flash decoding uses precomputed cos and sin for RoPE
            rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
                inference_context.max_sequence_length,
                self.rotary_pos_emb.get_cos_sin(inference_context.max_sequence_length),
            )
        else:
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_context, self.decoder, decoder_input, self.config, packed_seq_params
            )
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == 'thd',
            )
    elif self.position_embedding_type == 'mrope' and not self.config.multi_latent_attention:
        if self.training or not self.config.flash_decode:
            rotary_pos_emb = self.rotary_pos_emb(position_ids, self.mrope_section)
        else:
            # Flash decoding uses precomputed cos and sin for RoPE
            raise NotImplementedError(
                "Flash decoding uses precomputed cos and sin for RoPE, not implmented in "
                "MultimodalRotaryEmbedding yet."
            )

    detached_block_input = detach_tensor(decoder_input)

    # Run decoder.

    hidden_states, layer_graphs = transformer_block_forward(
        self.decoder,
        hidden_states=detached_block_input,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
        **(extra_block_kwargs or {}),
    )

    if not self.post_process:
        return hidden_states, ModelGraph(layer_graphs, hidden_states, preprocess_graph, detached_block_input)

    # logits and loss
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()
    logits, _ = self.output_layer(hidden_states, weight=output_weight)

    if labels is None:
        # [s b h] => [b s h]
        logits = logits.transpose(0, 1).contiguous()
        graph = ModelGraph(
            layer_graphs, hidden_states, preprocess_graph, detached_block_input
        )
        # print(f"in gpt_model_forward, return")
        return logits, graph

    loss = self.compute_language_model_loss(labels, logits)
    graph = ModelGraph(
        layer_graphs, hidden_states, preprocess_graph, detached_block_input
    )
    # print(f"in gpt_model_forward, return")
    return loss, graph


def gpt_model_backward(
    model_grad,
    model_graph: ModelGraph,
):  
    # print(f"in gpt_model_backward, start")
    block_input_grad = transformer_block_backward(model_grad, model_graph.layer_graphs)

    if model_graph.preprocess_graph[0] is not None:
        run_graph_backward(model_graph.preprocess_graph, block_input_grad, keep_graph=True, keep_grad=True)
        # print(f"in gpt_model_backward, return")
        return None
    else:
        # print(f"in gpt_model_backward, return")
        return block_input_grad


def gpt_model_forward_backward_overlaping(
    fwd_model,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_context: BaseInferenceContext = None,
    inference_params: InferenceParams = None,
    packed_seq_params: PackedSeqParams = None,
    extra_block_kwargs: dict = None,
):
    
    if extra_block_kwargs is None or extra_block_kwargs['bwd_model_graph'] is None:
        return gpt_model_forward(
            fwd_model, input_ids, position_ids, attention_mask, decoder_input, labels, inference_params,
            packed_seq_params, extra_block_kwargs
        )

    # print(f"in gpt_model_forward_backward_overlapping, start")
    bwd_model_grad, bwd_model_graph = extra_block_kwargs['bwd_model_grad'], extra_block_kwargs['bwd_model_graph']  # Fwd Model Decoder embedding.
    if decoder_input is not None:
        preprocess_graph = None
    elif fwd_model.pre_process:
        decoder_input = fwd_model.embedding(input_ids=input_ids, position_ids=position_ids)
        preprocess_graph = decoder_input
    else:
        # intermediate stage of pipeline
        # decoder will get hidden_states from encoder.input_tensor
        decoder_input = None
        preprocess_graph = None

    inference_context = deprecate_inference_params(inference_context, inference_params)

    # Rotary positional embeddings (embedding is None for PP intermediate devices)
    rotary_pos_emb = None
    rotary_pos_cos = None
    rotary_pos_sin = None
    if fwd_model.position_embedding_type == 'rope' and not fwd_model.config.multi_latent_attention:
        if not fwd_model.training and fwd_model.config.flash_decode and inference_context:
            assert (
                inference_context.is_static_batching()
            ), "GPTModel currently only supports static inference batching."
            # Flash decoding uses precomputed cos and sin for RoPE
            rotary_pos_cos, rotary_pos_sin = fwd_model.rotary_pos_emb_cache.setdefault(
                inference_context.max_sequence_length,
                fwd_model.rotary_pos_emb.get_cos_sin(inference_context.max_sequence_length),
            )
        else:
            rotary_seq_len = fwd_model.rotary_pos_emb.get_rotary_seq_len(
                inference_context, fwd_model.decoder, decoder_input, fwd_model.config, packed_seq_params
            )
            rotary_pos_emb = fwd_model.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == 'thd',
            )
    elif fwd_model.position_embedding_type == 'mrope' and not fwd_model.config.multi_latent_attention:
        if fwd_model.training or not fwd_model.config.flash_decode:
            rotary_pos_emb = fwd_model.rotary_pos_emb(position_ids, fwd_model.mrope_section)
        else:
            # Flash decoding uses precomputed cos and sin for RoPE
            raise NotImplementedError(
                "Flash decoding uses precomputed cos and sin for RoPE, not implmented in "
                "MultimodalRotaryEmbedding yet."
            )
    detached_block_input = detach_tensor(decoder_input)

    # Run transformer block fwd & bwd overlaping

    (hidden_states, layer_graphs), block_input_grad, pp_comm_output \
        = transformer_block_forward_backward_overlaping(
        fwd_model.decoder,
        detached_block_input,
        attention_mask,
        bwd_model_grad,
        bwd_model_graph.layer_graphs,
        rotary_pos_emb=rotary_pos_emb,
        inference_params=inference_params,
        packed_seq_params=packed_seq_params,
        pp_comm_params=extra_block_kwargs['pp_comm_params'],
        bwd_pp_comm_params=extra_block_kwargs['bwd_pp_comm_params']
    )

    if bwd_model_graph.preprocess_graph[0] is not None:
        run_graph_backward(bwd_model_graph.preprocess_graph, block_input_grad, keep_grad=True, keep_graph=True)

    if not fwd_model.post_process:
        return hidden_states, ModelGraph(layer_graphs, hidden_states, preprocess_graph,
                                         detached_block_input), pp_comm_output

    # logits and loss
    output_weight = None
    if fwd_model.share_embeddings_and_output_weights:
        output_weight = fwd_model.shared_embedding_or_output_weight()
    logits, _ = fwd_model.output_layer(hidden_states, weight=output_weight)

    if labels is None:
        # [s b h] => [b s h]
        logits = logits.transpose(0, 1).contiguous()
        graph = ModelGraph(
            layer_graphs, hidden_states, preprocess_graph, detached_block_input
        )
        # print(f"in gpt_model_forward_backward_overlaping, return")
        return logits, graph, pp_comm_output

    loss = fwd_model.compute_language_model_loss(labels, logits)
    graph = ModelGraph(
        layer_graphs, hidden_states, preprocess_graph, detached_block_input
    )
    # print(f"in gpt_model_forward_backward_overlaping, return")
    return loss, graph, pp_comm_output


