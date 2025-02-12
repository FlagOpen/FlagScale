import sys
import torch
sys.path.append("..")
from utils import padding_vocab_size


def get_hf_attn_ckpt(message, model, layer_id, args):
    nh = args.num_attention_heads
    ng = args.num_query_groups if args.group_query_attention else args.num_attention_heads
    dim = args.hidden_size
    assert nh % ng == 0

    tf_layer = model.model.layers[layer_id]
    
    message["q a weight"] = tf_layer.self_attn.q_a_proj.weight.data
    message["q a norm weight"] = tf_layer.self_attn.q_a_layernorm.weight.data
    message["q b weight"] = tf_layer.self_attn.q_b_proj.weight.data
    message["kv a weight"] = tf_layer.self_attn.kv_a_proj_with_mqa.weight.data
    message["kv a norm weight"] = tf_layer.self_attn.kv_a_layernorm.weight.data
    message["kv b weight"] = tf_layer.self_attn.kv_b_proj.weight.data
    message["o weight"] =  tf_layer.self_attn.o_proj.weight.data

    message["input norm weight"] = tf_layer.input_layernorm.weight.data
    message["post norm weight"] = tf_layer.post_attention_layernorm.weight.data


def get_hf_mlp_ckpt(message, model, layer_id, args):
    if layer_id < args.moe_num_first_k_dense_layers:
        get_hf_dense_mlp_ckpt(message, model, layer_id, args)
    else:
        get_hf_moe_mlp_ckpt(message, model, layer_id, args)



def get_hf_dense_mlp_ckpt(message, model, layer_id, args):
    tf_layer = model.model.layers[layer_id]
    
    message["gate weight"] = tf_layer.mlp.gate_proj.weight.data
    message["up weight"] = tf_layer.mlp.up_proj.weight.data
    message["down weight"] = tf_layer.mlp.down_proj.weight.data
    

def get_hf_moe_mlp_ckpt(message, model, layer_id, args):
    tf_layer = model.model.layers[layer_id]
    
    message["router weight"] = tf_layer.mlp.gate.weight.data
    message["router e score bias"] = tf_layer.mlp.gate.e_score_correction_bias.data
    message["shared expert gate weight"] = tf_layer.mlp.shared_experts.gate_proj.weight.data
    message["shared expert up weight"] = tf_layer.mlp.shared_experts.up_proj.weight.data
    message["shared expert down weight"] = tf_layer.mlp.shared_experts.down_proj.weight.data

    for id in range(args.num_experts):
        expert = tf_layer.mlp.experts[id]
        message[f"expert{id} gate weight"] = expert.gate_proj.weight.data
        message[f"expert{id} up weight"] = expert.up_proj.weight.data
        message[f"expert{id} down weight"] = expert.down_proj.weight.data
    
def get_hf_mtp_ckpt(message, model, mtp_layer_id, args):
    # Send transformer layers
    mtp_layer = model.model.layers[args.num_layers+mtp_layer_id]
    
    message["mtp word embeddings weight"] = mtp_layer.embed_tokens.weight.data
    message["mtp enorm weight"] = mtp_layer.enorm.weight.data
    message["mtp hnorm weight"] = mtp_layer.hnorm.weight.data
    message["mtp eh weight"] = mtp_layer.eh_proj.weight.data
    message["mtp shared head norm weight"] = mtp_layer.shared_head.norm.weight.data
    message["mtp shared head head weight"] = mtp_layer.shared_head.head.weight.data
    
    get_hf_attn_ckpt(message, model, args.num_layers+mtp_layer_id, args)
    get_hf_moe_mlp_ckpt(message, model, args.num_layers+mtp_layer_id, args)

def _get_parallel_size(args):
    return args.tensor_model_parallel_size, \
        args.pipeline_model_parallel_size, \
        args.expert_model_parallel_size, \
        args.virtual_pipeline_model_parallel_size or 1

def set_embedding_ckpt(message, models, md, args):
    tp_size, _, _, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"
    
    # embedding
    pos_embed = None
    if md.position_embedding_type == 'learned_absolute':
        pos_embed = message.pop("position embeddings")
    orig_word_embed = message.pop("word embeddings")
    full_word_embed = padding_vocab_size(orig_word_embed, md, args)

    # process world embedding in first pp stage
    out_word_embed = full_word_embed
    for tp_ep_rank, model in enumerate(models):
        model.embedding.word_embeddings.weight.data.copy_(out_word_embed)
        if pos_embed is not None:
            model.embedding.position_embeddings.weight.data.copy_(pos_embed)
        else:
            assert not hasattr(model.embedding, "position_embeddings")


def set_attn_ckpt(message, models, layer_id, md, args):
    tp_size, _, _, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"

    # weight
    q_a_weight = message.pop("q a weight")
    q_a_norm_weight = message.pop("q a norm weight")
    q_b_weight = message.pop("q b weight")
    kv_a_weight = message.pop("kv a weight")
    kv_a_norm_weight = message.pop("kv a norm weight")
    kv_b_weight = message.pop("kv b weight")
    o_weight = message.pop("o weight")
    input_norm_weight = message.pop("input norm weight")
    post_norm_weight = message.pop("post norm weight")

    # set data to transformer layer's self-attention
    for tp_ep_rank, model in enumerate(models):
        
        tf_layer = model.decoder.layers[layer_id]
        tf_layer.self_attention.linear_q_down_proj.weight.data.copy_(q_a_weight)
        tf_layer.self_attention.q_layernorm.weight.data.copy_(q_a_norm_weight)
        tf_layer.self_attention.linear_q_up_proj.weight.data.copy_(q_b_weight)
        tf_layer.self_attention.linear_kv_down_proj.weight.data.copy_(kv_a_weight)
        tf_layer.self_attention.kv_layernorm.weight.data.copy_(kv_a_norm_weight)
        tf_layer.self_attention.linear_kv_up_proj.weight.data.copy_(kv_b_weight)
        tf_layer.self_attention.linear_proj.weight.data.copy_(o_weight)
        tf_layer.input_layernorm.weight.data.copy_(input_norm_weight)
        tf_layer.pre_mlp_layernorm.weight.data.copy_(post_norm_weight)



def set_mlp_ckpt(message, model, layer_id, md, args):
    if layer_id < args.moe_num_first_k_dense_layers:
        set_dense_mlp_ckpt(message, model, layer_id, md, args)
    else:
        set_moe_mlp_ckpt(message, model, layer_id, md, args)



def set_dense_mlp_ckpt(message, models, layer_id, md, args):
    tp_size, _, ep_size, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"
    
    gate_weight = message.pop("gate weight")
    up_weight = message.pop("up weight")
    linear1_weight = torch.cat([gate_weight, up_weight], dim=0)
    linear2_weight = message.pop("down weight")
    
    for tp_ep_rank, model in enumerate(models):
        tf_layer = model.decoder.layers[layer_id]
        print(f"tf_layer.mlp.linear_fc1.weight shape is {tf_layer.mlp.linear_fc1.weight.shape}")
        tf_layer.mlp.linear_fc1.weight.data.copy_(linear1_weight)
        tf_layer.mlp.linear_fc2.weight.data.copy_(linear2_weight)


def set_moe_mlp_ckpt(message, models, layer_id, md, args):
    tp_size, _, ep_size, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"
    assert args.num_experts is not None, "deepseeks's num_experts cannot be None"
    
    assert md.previous_num_experts is not None
        
    # router
    router_weight = message.pop("router weight")
    router_score_bias = message.pop("router e score bias")
    
    # shared expert
    shared_expert_gate_weight = message.pop("shared expert gate weight")
    shared_expert_up_weight = message.pop("shared expert up weight")
    shared_expert_linear1_weight = torch.cat([shared_expert_gate_weight, shared_expert_up_weight], dim=0)
    shared_expert_linear2_weight = message.pop("shared expert down weight")
    
    # routed expert
    num_local_experts = md.previous_num_experts // ep_size
    for expert_id in range(num_local_experts):
        for ep_rank in range(ep_size):
            global_expert_id = ep_rank * num_local_experts + expert_id
            # weight
            gate_weight = message.pop(f"expert{global_expert_id} gate weight")
            up_weight = message.pop(f"expert{global_expert_id} up weight")
            linear1_weight = torch.cat([gate_weight, up_weight], dim=0)
            linear2_weight = message.pop(f"expert{global_expert_id} down weight")

            # set data
            for tp_rank in range(tp_size):
                tp_ep_rank = ep_rank * tp_size + tp_rank
                tf_layer = models[tp_ep_rank].decoder.layers[layer_id]
                # router
                router = tf_layer.mlp.router
                router.weight.data.copy_(router_weight)
                router.score_bias.data.copy_(router_score_bias)
                # shared expert
                shared_expert = tf_layer.mlp.shared_experts
                shared_expert.linear_fc1.weight.data.copy_(shared_expert_linear1_weight)
                shared_expert.linear_fc2.weight.data.copy_(shared_expert_linear2_weight)
                # routed expert
                expert = tf_layer.mlp.experts.local_experts[expert_id]
                expert.linear_fc1.weight.data.copy_(linear1_weight)
                expert.linear_fc2.weight.data.copy_(linear2_weight)


def set_final_norm_ckpt(message, models, md, args):
    final_norm_weight = message.pop("weight")
    for model in models:
        model.decoder.final_layernorm.weight.data.copy_(final_norm_weight)


def set_output_layer_ckpt(message, models, md, args):
    tp_size, _, _, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"
    
    orig_output_layer_weight = message.pop("weight")
    full_output_layer_weight = padding_vocab_size(orig_output_layer_weight, md, args)
    output_layer_weight = full_output_layer_weight
    for tp_ep_rank, model in enumerate(models):
        model.output_layer.weight.data.copy_(output_layer_weight)


def set_mtp_ckpt(message, models, md, mtp_layer_id, args):
    tp_size, _, _, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"

    mtp_layers = []
    for tp_ep_rank, model in enumerate(models):
        mtp_layer = model.mtp_predictor.mtp_modules[mtp_layer_id]
        mtp_layers.append(mtp_layer)

    # get and set transformer weights
    set_attn_ckpt(message, mtp_layers, 0, md, args)
    set_moe_mlp_ckpt(message, mtp_layers, 0, md, args)
        
    # get and set other weights
    mtp_embedding_weight = message.pop("mtp word embeddings weight")
    mtp_full_word_embedding_weight = padding_vocab_size(mtp_embedding_weight, md, args)
    mtp_enorm_weight = message.pop("mtp enorm weight")
    mtp_hnorm_weight = message.pop("mtp hnorm weight")
    mtp_eh_weight = message.pop("mtp eh weight")
    mtp_shared_head_norm_weight = message.pop("mtp shared head norm weight")
    mtp_shared_head_head_weight = message.pop("mtp shared head head weight")
    mtp_full_shared_head_head_weight = padding_vocab_size(mtp_shared_head_head_weight, md, args)
    for tp_ep_rank, model in enumerate(models):
        mtp_layer = model.mtp_predictor.mtp_modules[mtp_layer_id]
        mtp_layer.embedding.embedding.word_embeddings.weight.data.copy_(mtp_full_word_embedding_weight)
        mtp_layer.norm1.weight.data.copy_(mtp_enorm_weight)
        mtp_layer.norm2.weight.data.copy_(mtp_hnorm_weight)
        mtp_layer.linear_proj.weight.data.copy_(mtp_eh_weight)
        mtp_layer.decoder.final_layernorm.weight.data.copy_(mtp_shared_head_norm_weight)
        mtp_layer.output_head.head.weight.data.copy_(mtp_full_shared_head_head_weight)
    
        


