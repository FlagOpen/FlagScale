import torch
from tqdm import tqdm


def _set_preprocess_state(args, model, hf_model):
    '''Set embedding params.'''
    model.embedding.word_embeddings.weight.data.copy_(hf_model.model.embed_tokens.weight)


def _set_postprocess_state(args, model, hf_model):
    '''Set output layer & norm params.'''
    model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    model.output_layer.weight.data.copy_(hf_model.lm_head.weight)


def _set_attn_state(args, layer, hf_layer):
    '''Set self-attention params.'''

    # Get attention layer & state.
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    # Reshape loaded weights.
    nh = args.num_attention_heads
    ng = args.num_query_groups if args.group_query_attention else args.num_attention_heads
    dim = args.kv_channels
    assert nh % ng == 0

    # Copy weights (re-order dimensions for Megatron).
    attn.linear_qkv.weight.data.copy_(
        torch.cat([
            hf_attn.q_proj.weight.reshape((ng, dim*nh//ng, -1)),
            hf_attn.k_proj.weight.reshape((ng, dim, -1)),
            hf_attn.v_proj.weight.reshape((ng, dim, -1)),
        ], dim=1).reshape((-1, args.hidden_size))
    )
    attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)
    attn.linear_qkv.layer_norm_weight.data.copy_(hf_layer.post_attention_layernorm.weight)


def _set_mlp_state(args, layer, hf_layer):
    '''Set MLP params.'''

    mlp = layer.mlp
    hf_mlp = hf_layer.block_sparse_moe

    mlp.router.weight = hf_mlp.gate.weight
    for id in range(args.num_experts):
        expert = mlp.experts.local_experts[id]
        hf_expert = hf_mlp.experts[id]

        expert.linear_fc1.weight.data.copy_(
            torch.cat([
                hf_expert.w1.weight,
                hf_expert.w3.weight,
            ], dim=0)
        )
        expert.linear_fc2.weight.data.copy_(
            hf_expert.w2.weight
        )


def _set_layer_state(args, model, hf_model, layer_idx):
    '''Set transformer layer params.'''

    layer = model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    _set_attn_state(args, layer, hf_layer)
    _set_mlp_state(args, layer, hf_layer)
    layer.pre_mlp_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight)


def load_checkpoint_hf2mg(args):
    '''Set model params.'''
    from .model import get_hf_model, get_mg_model

    hf_model = get_hf_model(args.load)
    model = get_mg_model(args.params_dtype, True, True)

    # Set model state.
    _set_preprocess_state(args, model, hf_model)
    _set_postprocess_state(args, model, hf_model)
    for layer_idx in tqdm(range(args.num_layers), "set layer states"):
        _set_layer_state(args, model, hf_model, layer_idx)
    model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)

    return model, hf_model
