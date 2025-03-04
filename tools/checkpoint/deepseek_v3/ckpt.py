import sys

import torch

sys.path.append("..")
from utils import padding_vocab_size


def _get_parallel_size(args):
    return (
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.expert_model_parallel_size,
        args.virtual_pipeline_model_parallel_size or 1,
    )


#### load from huggingface ckpt
def get_hf_attn_ckpt(message, model, layer_id, args):
    nh = args.num_attention_heads
    ng = (
        args.num_query_groups
        if args.group_query_attention
        else args.num_attention_heads
    )
    dim = args.hidden_size
    assert nh % ng == 0

    tf_layer = model.model.layers[layer_id]

    if args.q_lora_rank is not None:
        message["q a weight"] = tf_layer.self_attn.q_a_proj.weight.data
        message["q a norm weight"] = tf_layer.self_attn.q_a_layernorm.weight.data
        message["q b weight"] = tf_layer.self_attn.q_b_proj.weight.data
    else:
        message["q weight"] = tf_layer.self_attn.q_proj.weight.data

    message["kv a weight"] = tf_layer.self_attn.kv_a_proj_with_mqa.weight.data
    message["kv a norm weight"] = tf_layer.self_attn.kv_a_layernorm.weight.data
    message["kv b weight"] = tf_layer.self_attn.kv_b_proj.weight.data
    message["o weight"] = tf_layer.self_attn.o_proj.weight.data
    message["input norm weight"] = tf_layer.input_layernorm.weight.data
    message["post norm weight"] = tf_layer.post_attention_layernorm.weight.data


def get_hf_mlp_ckpt(message, model, layer_id, args):
    first_k_dense_replace = args.moe_layer_freq.index(1)
    if layer_id < first_k_dense_replace:
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
    if hasattr(tf_layer.mlp.gate, "e_score_correction_bias"):
        message["router e score bias"] = tf_layer.mlp.gate.e_score_correction_bias.data
    message["shared expert gate weight"] = (
        tf_layer.mlp.shared_experts.gate_proj.weight.data
    )
    message["shared expert up weight"] = tf_layer.mlp.shared_experts.up_proj.weight.data
    message["shared expert down weight"] = (
        tf_layer.mlp.shared_experts.down_proj.weight.data
    )

    for id in range(args.num_experts):
        expert = tf_layer.mlp.experts[id]
        message[f"expert{id} gate weight"] = expert.gate_proj.weight.data
        message[f"expert{id} up weight"] = expert.up_proj.weight.data
        message[f"expert{id} down weight"] = expert.down_proj.weight.data


def get_hf_mtp_ckpt(message, model, mtp_layer_id, args):
    # Send transformer layers
    mtp_layer = model.model.layers[args.num_layers + mtp_layer_id]

    message["mtp word embeddings weight"] = mtp_layer.embed_tokens.weight.data
    message["mtp enorm weight"] = mtp_layer.enorm.weight.data
    message["mtp hnorm weight"] = mtp_layer.hnorm.weight.data
    message["mtp eh weight"] = mtp_layer.eh_proj.weight.data
    message["mtp shared head norm weight"] = mtp_layer.shared_head.norm.weight.data
    message["mtp shared head head weight"] = mtp_layer.shared_head.head.weight.data

    get_hf_attn_ckpt(message, model, args.num_layers + mtp_layer_id, args)
    get_hf_moe_mlp_ckpt(message, model, args.num_layers + mtp_layer_id, args)


#### set to megatron ckpt
def set_embedding_ckpt(message, models, md, args):
    tp_size, _, _, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"

    # embedding
    pos_embed = None
    if md.position_embedding_type == "learned_absolute":
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
    if args.q_lora_rank is not None:
        q_a_weight = message.pop("q a weight")
        q_a_norm_weight = message.pop("q a norm weight")
        q_b_weight = message.pop("q b weight")
    else:
        q_weight = message.pop("q weight")

    kv_a_weight = message.pop("kv a weight")
    kv_a_norm_weight = message.pop("kv a norm weight")
    kv_b_weight = message.pop("kv b weight")
    o_weight = message.pop("o weight")
    input_norm_weight = message.pop("input norm weight")

    first_k_dense_replace = args.moe_layer_freq.index(1)
    if args.total_layer_num >= first_k_dense_replace:
        post_norm_weight = message.pop("post norm weight")

    # set data to transformer layer's self-attention
    for tp_ep_rank, model in enumerate(models):
        if hasattr(model, "decoder"):
            tf_layer = model.decoder.layers[layer_id]
        else:
            tf_layer = model.transformer_layer  # for mtp
        if args.q_lora_rank is not None:
            tf_layer.self_attention.linear_q_down_proj.weight.data.copy_(q_a_weight)
            tf_layer.self_attention.linear_q_up_proj.layer_norm_weight.data.copy_(
                q_a_norm_weight
            )
            tf_layer.self_attention.linear_q_up_proj.weight.data.copy_(q_b_weight)
        else:
            tf_layer.self_attention.linear_q_proj.weight.data.copy_(q_weight)

        tf_layer.self_attention.linear_kv_down_proj.weight.data.copy_(kv_a_weight)
        tf_layer.self_attention.linear_kv_up_proj.layer_norm_weight.data.copy_(
            kv_a_norm_weight
        )
        tf_layer.self_attention.linear_kv_up_proj.weight.data.copy_(kv_b_weight)
        tf_layer.self_attention.linear_proj.weight.data.copy_(o_weight)
        tf_layer.input_layernorm.weight.data.copy_(input_norm_weight)

        if args.total_layer_num >= first_k_dense_replace:
            tf_layer.pre_mlp_layernorm.weight.data.copy_(post_norm_weight)


def set_mlp_ckpt(message, model, layer_id, md, args):
    first_k_dense_replace = args.moe_layer_freq.index(1)
    if args.total_layer_num < first_k_dense_replace:
        set_dense_mlp_ckpt(message, model, layer_id, md, args)
    else:
        set_moe_mlp_ckpt(message, model, layer_id, md, args)


def set_dense_mlp_ckpt(message, models, layer_id, md, args):
    tp_size, _, ep_size, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"

    post_norm_weight = message.pop("post norm weight")
    gate_weight = message.pop("gate weight")
    up_weight = message.pop("up weight")
    linear1_weight = torch.cat([gate_weight, up_weight], dim=0)
    linear2_weight = message.pop("down weight")

    for tp_ep_rank, model in enumerate(models):
        if hasattr(model, "decoder"):
            tf_layer = model.decoder.layers[layer_id]
        else:
            tf_layer = model.transformer_layer  # for mtp
        tf_layer.mlp.linear_fc1.layer_norm_weight.data.copy_(post_norm_weight)
        tf_layer.mlp.linear_fc1.weight.data.copy_(linear1_weight)
        tf_layer.mlp.linear_fc2.weight.data.copy_(linear2_weight)


def set_moe_mlp_ckpt(message, models, layer_id, md, args):
    tp_size, _, ep_size, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"
    assert args.num_experts is not None, "deepseeks's num_experts cannot be None"

    assert md.previous_num_experts is not None

    # router
    router_weight = message.pop("router weight")
    router_score_bias = None
    use_router_score_bias = False
    if "router e score bias" in message.keys():
        use_router_score_bias = True
        router_score_bias = message.pop("router e score bias")

    # shared expert
    shared_expert_gate_weight = message.pop("shared expert gate weight")
    shared_expert_up_weight = message.pop("shared expert up weight")
    shared_expert_linear1_weight = torch.cat(
        [shared_expert_gate_weight, shared_expert_up_weight], dim=0
    )
    shared_expert_linear2_weight = message.pop("shared expert down weight")

    # if not args.moe_grouped_gemm:
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
                if hasattr(models[tp_ep_rank], "decoder"):
                    tf_layer = models[tp_ep_rank].decoder.layers[layer_id]
                else:
                    tf_layer = models[tp_ep_rank].transformer_layer  # for mtp
                # router
                router = tf_layer.mlp.router
                router.weight.data.copy_(router_weight)
                if use_router_score_bias:
                    router.score_bias.data.copy_(router_score_bias)
                # shared expert
                shared_expert = tf_layer.mlp.shared_experts
                shared_expert.linear_fc1.weight.data.copy_(shared_expert_linear1_weight)
                shared_expert.linear_fc2.weight.data.copy_(shared_expert_linear2_weight)
                # routed expert
                if not args.moe_grouped_gemm:
                    expert = tf_layer.mlp.experts.local_experts[expert_id]
                    expert.linear_fc1.weight.data.copy_(linear1_weight)
                    expert.linear_fc2.weight.data.copy_(linear2_weight)
                else:  # using TEGroupedMLP
                    expert_linear_fc1_weight = getattr(
                        tf_layer.mlp.experts.linear_fc1, f"weight{expert_id}", None
                    )
                    expert_linear_fc2_weight = getattr(
                        tf_layer.mlp.experts.linear_fc2, f"weight{expert_id}", None
                    )
                    expert_linear_fc1_weight.data.copy_(linear1_weight)
                    expert_linear_fc2_weight.data.copy_(linear2_weight)


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
    # mtp embeddings weight is shared with main model embeddings
    mtp_embeddings_weight = message.pop("mtp word embeddings weight")
    full_word_embed = padding_vocab_size(mtp_embeddings_weight, md, args)
    mtp_enorm_weight = message.pop("mtp enorm weight")
    mtp_hnorm_weight = message.pop("mtp hnorm weight")
    mtp_eh_weight = message.pop("mtp eh weight")
    mtp_shared_head_norm_weight = message.pop("mtp shared head norm weight")
    for tp_ep_rank, model in enumerate(models):
        model.mtp_embedding.weight.data.copy_(full_word_embed)
        mtp_layer = model.mtp_predictor.mtp_modules[mtp_layer_id]
        mtp_layer.norm1.weight.data.copy_(mtp_enorm_weight)
        mtp_layer.norm2.weight.data.copy_(mtp_hnorm_weight)
        mtp_layer.linear_proj.weight.data.copy_(mtp_eh_weight)
        mtp_layer.final_norm.weight.data.copy_(mtp_shared_head_norm_weight)
    # mtp output lm head is the same with main model output lm head


#### load from megatron ckpt
def get_embedding_ckpt(message, models, args):
    tp_size, _, _, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"

    word_embeddings = None
    complete_tp_ranks = []
    for tp_ep_rank, model in enumerate(models):
        tp_rank = tp_ep_rank % tp_size
        if tp_rank in complete_tp_ranks:
            continue
        complete_tp_ranks.append(tp_rank)
        word_embeddings = model.embedding.word_embeddings.weight.data
    message["word embeddings"] = word_embeddings


def get_attn_ckpt(message, models, layer_id, args):
    tp_size, _, _, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"

    # parallel tensor
    q_a_weight = None
    q_a_norm_weight = None
    q_b_weight = None
    q_weight = None
    kv_a_weight = None
    kv_a_norm_weight = None
    kv_b_weight = None
    o_weight = None
    # non-parallel tensor
    input_norm_weight = None
    post_norm_weight = None

    first_k_dense_replace = args.moe_layer_freq.index(1)
    complete_tp_ranks = []
    for tp_ep_rank, model in enumerate(models):
        tp_rank = tp_ep_rank % tp_size
        if tp_rank in complete_tp_ranks:
            continue
        complete_tp_ranks.append(tp_rank)

        if hasattr(model, "decoder"):
            tf_layer = model.decoder.layers[layer_id]
        else:
            tf_layer = model.transformer_layer  # for mtp
        # weight
        if args.q_lora_rank is not None:
            q_a_weight = tf_layer.self_attention.linear_q_down_proj.weight.data
            q_a_norm_weight = (
                tf_layer.self_attention.linear_q_up_proj.layer_norm_weight.data
            )
            q_b_weight = tf_layer.self_attention.linear_q_up_proj.weight.data
        else:
            q_weight = tf_layer.self_attention.linear_q_proj.weight.data

        kv_a_weight = tf_layer.self_attention.linear_kv_down_proj.weight.data
        kv_a_norm_weight = (
            tf_layer.self_attention.linear_kv_up_proj.layer_norm_weight.data
        )
        kv_b_weight = tf_layer.self_attention.linear_kv_up_proj.weight.data
        o_weight = tf_layer.self_attention.linear_proj.weight.data
        input_norm_weight = tf_layer.input_layernorm.weight.data

        if args.total_layer_num >= first_k_dense_replace:
            post_norm_weight = tf_layer.pre_mlp_layernorm.weight.data

    # weight
    if args.q_lora_rank is not None:
        message["q a weight"] = q_a_weight
        message["q a norm weight"] = q_a_norm_weight
        message["q b weight"] = q_b_weight
    else:
        message["q weight"] = q_weight

    message["kv a weight"] = kv_a_weight
    message["kv a norm weight"] = kv_a_norm_weight
    message["kv b weight"] = kv_b_weight
    message["o weight"] = o_weight
    message["input norm weight"] = input_norm_weight
    if args.total_layer_num >= first_k_dense_replace:
        message["post norm weight"] = post_norm_weight


def get_mlp_ckpt(message, models, layer_id, args):
    first_k_dense_replace = args.moe_layer_freq.index(1)
    if args.total_layer_num < first_k_dense_replace:
        get_dense_mlp_ckpt(message, models, layer_id, args)
    else:
        get_moe_mlp_ckpt(message, models, layer_id, args)


def get_dense_mlp_ckpt(message, models, layer_id, args):
    tp_size, _, _, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"

    # parallel tensor
    post_norm_weight = None
    gate_weight = None
    up_weight = None
    down_weight = None

    complete_tp_ranks = []
    for tp_ep_rank, model in enumerate(models):
        tp_rank = tp_ep_rank % tp_size
        if tp_rank in complete_tp_ranks:
            continue
        complete_tp_ranks.append(tp_rank)

        if hasattr(model, "decoder"):
            tf_layer = model.decoder.layers[layer_id]
        else:
            tf_layer = model.transformer_layer  # for mtp
        post_norm_weight = tf_layer.mlp.linear_fc1.layer_norm_weight.data
        gate_weight = tf_layer.mlp.linear_fc1.weight.data[0]
        up_weight = tf_layer.mlp.linear_fc1.weight.data[1]
        down_weight = tf_layer.mlp.linear_fc2.weight.data

    # weight
    message["post norm weight"] = post_norm_weight
    message["gate weight"] = gate_weight
    message["up weight"] = up_weight
    message["down weight"] = down_weight


def get_moe_mlp_ckpt(message, models, layer_id, args):
    tp_size, _, ep_size, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"

    assert args.num_experts is not None and args.num_experts % ep_size == 0
    num_local_experts = args.num_experts // ep_size
    for expert_id in range(num_local_experts):
        for ep_rank in range(ep_size):
            global_expert_id = num_local_experts * ep_rank + expert_id

            # local experts
            use_router_score_bias = False
            for tp_rank in range(tp_size):
                tp_ep_rank = ep_rank * tp_size + tp_rank
                if hasattr(models[tp_ep_rank], "decoder"):
                    tf_layer = models[tp_ep_rank].decoder.layers[layer_id]
                else:
                    tf_layer = models[tp_ep_rank].transformer_layer  # for mtp

                # router
                router = tf_layer.mlp.router
                router_weight = router.weight.data
                if hasattr(router, "score_bias"):
                    use_router_score_bias = True
                    router_score_bias = router.score_bias.data
                # shared experts
                shared_expert = tf_layer.mlp.shared_experts
                shared_expert_gate_weight = shared_expert.linear_fc1.weight.data[0]
                shared_expert_up_weight = shared_expert.linear_fc1.weight.data[1]
                shared_expert_down_weight = shared_expert.linear_fc2.weight.data
                # routed experts
                if not args.moe_grouped_gemm:
                    expert = tf_layer.mlp.experts.local_experts[expert_id]
                    expert_gate_weight = expert.linear_fc1.weight.data[0]
                    expert_up_weight = expert.linear_fc1.weight.data[1]
                    expert_down_weight = expert.lienar_fc2.weight.data
                else:  # using TEGroupedMLP
                    expert_linear_fc1_weight = getattr(
                        tf_layer.mlp.experts.linear_fc1, f"weight{expert_id}", None
                    )
                    expert_gate_weight = expert_linear_fc1_weight.data[0]
                    expert_up_weight = expert_linear_fc1_weight.data[1]
                    expert_down_weight = getattr(
                        tf_layer.mlp.experts.linear_fc2, f"weight{expert_id}", None
                    )

            message["router weight"] = router_weight
            if use_router_score_bias:
                message["router e score bias"] = router_score_bias
            message["shared expert gate weight"] = shared_expert_gate_weight
            message["shared expert up weight"] = shared_expert_up_weight
            message["shared expert down weight"] = shared_expert_down_weight

            message[f"expert{global_expert_id} gate weight"] = expert_gate_weight
            message[f"expert{global_expert_id} up weight"] = expert_up_weight
            message[f"expert{global_expert_id} down weight"] = expert_down_weight


def get_final_norm_ckpt(message, models, args):
    tp_size, _, _, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"

    final_layernorm_weight = None
    complete_tp_ranks = []
    for tp_ep_rank, model in enumerate(models):
        tp_rank = tp_ep_rank % tp_size
        if tp_rank in complete_tp_ranks:
            continue
        complete_tp_ranks.append(tp_rank)
        final_layernorm_weight = model.decoder.final_layernorm.weight.data

    message["weight"] = final_layernorm_weight


def get_output_layer_ckpt(message, models, args):
    tp_size, _, _, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"

    output_layer_weight = None
    complete_tp_ranks = []
    for tp_ep_rank, model in enumerate(models):
        tp_rank = tp_ep_rank % tp_size
        if tp_rank in complete_tp_ranks:
            continue
        complete_tp_ranks.append(tp_rank)
        output_layer_weight = model.output_layer.weight.data
    message["weight"] = output_layer_weight


def get_mtp_ckpt(message, models, mtp_layer_id, args):
    tp_size, _, _, _ = _get_parallel_size(args)
    assert tp_size == 1, "do not support TP parallel for deepseek v3 currently"

    mtp_layers = []
    for tp_ep_rank, model in enumerate(models):
        mtp_layer = model.mtp_predictor.mtp_modules[mtp_layer_id]
        mtp_layers.append(mtp_layer)

    # get and set transformer weights
    get_attn_ckpt(message, mtp_layers, 0, args)
    get_moe_mlp_ckpt(message, mtp_layers, 0, args)

    complete_tp_ranks = []
    for tp_ep_rank, model in enumerate(models):
        tp_rank = tp_ep_rank % tp_size
        if tp_rank in complete_tp_ranks:
            continue
        complete_tp_ranks.append(tp_rank)

        mtp_word_embedding_weight = model.mtp_embedding.word_embeddings.weight.data
        mtp_layer = model.mtp_predictor.mtp_modules[mtp_layer_id]
        mtp_enorm_weight = mtp_layer.norm1.weight.data
        mtp_hnorm_weight = mtp_layer.norm2.weight.data
        mtp_eh_weight = mtp_layer.linear_proj.weight.data
        mtp_norm_weight = mtp_layer.final_norm.weight.data
        output_layer_weight = model.output_layer.weight.data

    message["mtp word embeddings weight"] = mtp_word_embedding_weight
    message["mtp enorm weight"] = mtp_enorm_weight
    message["mtp hnorm weight"] = mtp_hnorm_weight
    message["mtp eh weight"] = mtp_eh_weight
    message["mtp shared head norm weight"] = mtp_norm_weight
    message["mtp shared head head weight"] = output_layer_weight


#### set to huggingface ckpt
def set_hf_attn_ckpt(message, model, layer_id, md, args):
    if args.q_lora_rank is not None:
        q_a_weight = message.pop("q a weight")
        q_a_norm_weight = message.pop("q a norm weight")
        q_b_weight = message.pop("q b weight")
    else:
        q_weight = message.pop("q weight")
    kv_a_weight = message.pop("kv a weight")
    kv_a_norm_weight = message.pop("kv a norm weight")
    kv_b_weight = message.pop("kv b weight")
    o_weight = message.pop("o weight")
    input_norm_weight = message.pop("input norm weight")
    first_k_dense_replace = args.moe_layer_freq.index(1)
    if args.total_layer_num >= first_k_dense_replace:
        post_norm_weight = message.pop("post norm weight")

    tf_layer = model.model.layers[layer_id]
    if args.q_lora_rank is not None:
        tf_layer.self_attn.q_a_proj.weight.data.copy_(q_a_weight)
        tf_layer.self_attn.q_a_layernorm.weight.data.copy_(q_a_norm_weight)
        tf_layer.self_attn.q_b_proj.weight.data.copy_(q_b_weight)
    else:
        tf_layer.self_attn.q_proj.weight.data.copy_(q_weight)
    tf_layer.self_attn.kv_a_proj_with_mqa.weight.data.copy_(kv_a_weight)
    tf_layer.self_attn.kv_a_layernorm.weight.data.copy_(kv_a_norm_weight)
    tf_layer.self_attn.kv_b_proj.weight.data.copy_(kv_b_weight)
    tf_layer.self_attn.o_proj.weight.data.copy_(o_weight)
    tf_layer.input_layernorm.weight.data.copy_(input_norm_weight)
    first_k_dense_replace = args.moe_layer_freq.index(1)
    if args.total_layer_num >= first_k_dense_replace:
        tf_layer.post_attention_layernorm.weight.data.copy_(post_norm_weight)


def set_hf_mlp_ckpt(message, model, layer_id, md, args):
    first_k_dense_replace = args.moe_layer_freq.index(1)
    if args.total_layer_num < first_k_dense_replace:
        set_hf_dense_mlp_ckpt(message, model, layer_id, md, args)
    else:
        set_hf_moe_mlp_ckpt(message, model, layer_id, md, args)


def set_hf_dense_mlp_ckpt(message, model, layer_id, md, args):
    post_norm_weight = message.pop("post norm weight")
    gate_weight = message.pop("gate weight")
    up_weight = message.pop("up weight")
    down_weight = message.pop("down weight")

    tf_layer = model.model.layers[layer_id]
    tf_layer.post_attention_layernorm.weight.data.copy_(post_norm_weight)
    tf_layer.mlp.gate_proj.weight.data.copy_(gate_weight)
    tf_layer.mlp.up_proj.weight.data.copy_(up_weight)
    tf_layer.mlp.down_proj.weight.data.copy_(down_weight)


def set_hf_moe_mlp_ckpt(message, model, layer_id, md, args):
    tf_layer = model.model.layers[layer_id]

    router_weight = message.pop("router weight")
    use_router_score_bias = False
    if "router e score bias" in message.keys():
        use_router_score_bias = True
        router_score_bias = message.pop("router e score bias")
    shared_expert_gate_weight = message.pop("shared expert gate weight")
    shared_expert_up_weight = message.pop("shared expert up weight")
    shared_expert_down_weight = message.pop("shared expert down weight")

    tf_layer.mlp.gate.weight.data.copy_(router_weight)
    if use_router_score_bias:
        tf_layer.mlp.gate.e_score_correction_bias.data.copy_(router_score_bias)
    tf_layer.mlp.shared_experts.gate_proj.weight.data.copy_(shared_expert_gate_weight)
    tf_layer.mlp.shared_experts.up_proj.weight.data.copy_(shared_expert_up_weight)
    tf_layer.mlp.shared_experts.down_proj.weight.data.copy_(shared_expert_down_weight)

    for global_expert_id in range(args.num_experts):
        expert_gate_weight = message.pop(f"expert{global_expert_id} gate weight")
        expert_up_weight = message.pop(f"expert{global_expert_id} up weight")
        expert_down_weight = message.pop(f"expert{global_expert_id} down weight")
        tf_layer.mlp.experts[global_expert_id].gate_proj.weight.data.copy_(
            expert_gate_weight
        )
        tf_layer.mlp.experts[global_expert_id].up_proj.weight.data.copy_(
            expert_up_weight
        )
        tf_layer.mlp.experts[global_expert_id].down_proj.weight.data.copy_(
            expert_down_weight
        )


def set_hf_mtp_ckpt(message, model, mtp_layer_id, md, args):
    layer_id = args.num_layers + mtp_layer_id
    set_hf_attn_ckpt(message, model, layer_id, md, args)
    set_hf_mlp_ckpt(message, model, layer_id, md, args)

    mtp_word_embedding_weight = message.pop("mtp word embeddings weight")
    print("Warning: saver_transformers will change embedding to be no-padded .")
    full_word_embed = padding_vocab_size(mtp_word_embedding_weight, md, args)[
        : args.vocab_size, :
    ]
    mtp_enorm_weight = message.pop("mtp enorm weight")
    mtp_hnorm_weight = message.pop("mtp hnorm weight")
    mtp_eh_weight = message.pop("mtp eh weight")
    mtp_norm_weight = message.pop("mtp shared head norm weight")
    output_layer_weight = message.pop("mtp shared head head weight")
    print("Warning: saver_transformers will change output_layer to be no-padded .")
    full_output_layer_weight = padding_vocab_size(output_layer_weight, md, args)[
        : args.vocab_size, :
    ]

    tf_layer = model.model.layers[layer_id]
    tf_layer.embed_tokens.weight.data.copy_(full_word_embed)
    tf_layer.enorm.weight.data.copy_(mtp_enorm_weight)
    tf_layer.hnorm.weight.data.copy_(mtp_hnorm_weight)
    tf_layer.eh_proj.weight.data.copy_(mtp_eh_weight)
    tf_layer.shared_head.norm.weight.data.copy_(mtp_norm_weight)
    tf_layer.shared_head.head.weight.data.copy_(full_output_layer_weight)
