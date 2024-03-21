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

    # Copy bias
    if args.add_qkv_bias or args.add_bias_linear:
        attn.linear_qkv.bias.data.copy_(
            torch.cat([
                hf_attn.q_proj.bias.reshape((ng, dim*nh//ng, -1)),
                hf_attn.k_proj.bias.reshape((ng, dim, -1)),
                hf_attn.v_proj.bias.reshape((ng, dim, -1)),
            ], dim=1).reshape((-1))
        )

    if args.add_bias_linear:
        attn.linear_proj.bias.data.copy_(hf_attn.o_proj.bias)


def _set_mlp_state(args, layer, hf_layer):
    '''Set MLP params.'''

    mlp = layer.mlp
    hf_mlp = hf_layer.mlp

    # Copy weight
    mlp.linear_fc1.layer_norm_weight.data.copy_(hf_layer.input_layernorm.weight)
    mlp.linear_fc1.weight.data.copy_(
        torch.cat([
                hf_mlp.gate_proj.weight,
                hf_mlp.up_proj.weight,
        ], dim=0)
    )
    mlp.linear_fc2.weight.data.copy_(hf_mlp.down_proj.weight)

    # Copy bias
    if args.add_bias_linear:
        mlp.linear_fc1.bias.data.copy_(
            torch.cat([
                    hf_mlp.gate_proj.bias,
                    hf_mlp.up_proj.bias,
            ], dim=0)
        )
        mlp.linear_fc2.bias.data.copy_(hf_mlp.down_proj.bias)


def _set_layer_state(args, model, hf_model, layer_idx):
    '''Set transformer layer params.'''

    layer = model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    _set_attn_state(args, layer, hf_layer)
    _set_mlp_state(args, layer, hf_layer)


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

    return model


def _get_parallel_size(args):
    assert args.expert_model_parallel_size == 1
    return args.tensor_model_parallel_size, \
        args.pipeline_model_parallel_size, \
        args.expert_model_parallel_size, \
        args.virtual_pipeline_model_parallel_size or 1


def get_attn_ckpt(message, models, layer_id, margs):
    tp_size, _, _, _ = _get_parallel_size(margs)

    # parallel tensor
    qkv_weight = []
    qkv_bias = []
    proj_weight = []
    # non-parallel tensor
    post_norm_weight = None
    post_norm_bias = None
    proj_bias = None

    assert len(models) == tp_size
    for model in models:
        tf_layer = model.decoder.layers[layer_id]
        # weight
        qkv_weight.append(tf_layer.self_attention.linear_qkv.weight.data)
        proj_weight.append(tf_layer.self_attention.linear_proj.weight.data)
        post_norm_weight = tf_layer.self_attention.linear_qkv.layer_norm_weight.data
        # bias
        if margs.norm_has_bias:
            post_norm_bias = tf_layer.self_attention.linear_qkv.layer_norm_bias.data
        if margs.add_qkv_bias or margs.add_bias_linear:
            qkv_bias.append(tf_layer.self_attention.linear_qkv.bias.data)
        if margs.add_bias_linear:
            proj_bias = tf_layer.self_attention.linear_proj.bias.data

    # weight
    message["qkv weight"] = torch.cat(qkv_weight, dim=0)
    message["proj weight"] = torch.cat(proj_weight, dim=1)
    message["post norm weight"] = post_norm_weight
    # bias
    if margs.norm_has_bias:
        message["post norm bias"] = post_norm_bias
    if margs.add_qkv_bias or margs.add_bias_linear:
        message["qkv bias"] = torch.cat(qkv_bias, dim=0)
    if margs.add_bias_linear:
        message["proj bias"] = proj_bias


def get_mlp_ckpt(message, models, layer_id, margs):
    tp_size, _, _, _ = _get_parallel_size(margs)

    # parallel tensor
    l0_weight = []
    l0_bias = []
    l1_weight = []
    # non-parallel tensor
    l1_bias = None
    pre_norm_weight = None
    pre_norm_bias = None

    assert len(models) == tp_size
    for model in models:
        tf_layer = model.decoder.layers[layer_id]
        # weight
        l0_weight.append(tf_layer.mlp.linear_fc1.weight.data)
        l1_weight.append(tf_layer.mlp.linear_fc2.weight.data)
        pre_norm_weight = tf_layer.mlp.linear_fc1.layer_norm_weight.data
        # bias
        if margs.norm_has_bias:
            pre_norm_bias = tf_layer.mlp.linear_fc1.layer_norm_bias.data
        if margs.add_bias_linear:
            l0_bias.append(tf_layer.mlp.linear_fc1.bias.data)
            l1_bias = tf_layer.mlp.linear_fc2.bias.data

    # weight
    message["pre norm weight"] = pre_norm_weight
    message["mlp l1 weight"] = torch.cat(l1_weight, dim=1)
    if margs.swiglu:
        for tp_rank in range(tp_size):
            l0_weight[tp_rank] = torch.chunk(l0_weight[tp_rank], 2, dim=0)
        message["mlp l0 weight W"] = torch.cat([w[0] for w in l0_weight], dim=0)
        message["mlp l0 weight V"] = torch.cat([w[1] for w in l0_weight], dim=0)
    else:
        message["mlp l0 weight"] = torch.cat(l0_weight, dim=0)
    # bias
    if margs.norm_has_bias:
        message["pre norm bias"] = pre_norm_bias
    if margs.add_bias_linear:
        message["mlp l1 bias"] = l1_bias
        if margs.swiglu:
            for tp_rank in range(tp_size):
                l0_bias[tp_rank] = torch.chunk(l0_bias[tp_rank], 2, dim=0)
            message["mlp l0 bias W"] = torch.cat([b[0] for b in l0_bias],dim=0)
            message["mlp l0 bias V"] = torch.cat([b[1] for b in l0_bias],dim=0)
        else:
            message["mlp l0 bias"] = torch.cat(l0_bias, dim=0)


def set_attn_ckpt(message, models, layer_id, md, margs):
    tp_size, _, _, _ = _get_parallel_size(margs)

    # weight
    qkv_weight = torch.chunk(message.pop("qkv weight"), tp_size, dim=0)
    proj_weight = torch.chunk(message.pop("proj weight"), tp_size, dim=1)
    post_norm_weight = message.pop("post norm weight")
    # bias
    if md.norm_has_bias:
        post_norm_bias = message.pop("post norm bias")
    if md.add_qkv_bias or md.add_bias_linear:
        qkv_bias = torch.chunk(message.pop("qkv bias"), tp_size, dim=0)
    if md.add_bias_linear:
        proj_bias = message.pop("proj bias")

    # set data to transformer layer's self-attention
    for tp_rank, model in enumerate(models):
        layer = model.decoder.layers[layer_id]
        layer.self_attention.linear_qkv.weight.data.copy_(qkv_weight[tp_rank])
        layer.self_attention.linear_proj.weight.data.copy_(proj_weight[tp_rank])
        layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(post_norm_weight)
        if md.norm_has_bias:
            layer.self_attention.linear_qkv.layer_norm_bias.data.copy_(post_norm_bias)
        if md.add_qkv_bias or md.add_bias_linear:
            layer.self_attention.linear_qkv.bias.data.copy_(qkv_bias[tp_rank])
        if md.add_bias_linear:
            layer.self_attention.linear_proj.bias.data.copy_(proj_bias)


def set_mlp_ckpt(message, models, layer_id, md, margs):
    tp_size, _, _, _ = _get_parallel_size(margs)

    # weight
    pre_norm_weight = message.pop("pre norm weight")
    l1_weight = torch.chunk(message.pop("mlp l1 weight"), tp_size, dim=1)
    if md.swiglu:
        l0_weight_W = torch.chunk(message.pop("mlp l0 weight W"), tp_size, dim=0)
        l0_weight_V = torch.chunk(message.pop("mlp l0 weight V"), tp_size, dim=0)
        l0_weight = [torch.cat(weights, dim=0) for weights in zip(l0_weight_W, l0_weight_V)]
    else:
        l0_weight = torch.chunk(message.pop("mlp l0 weight"), tp_size, dim=0)
    # bias
    if md.norm_has_bias:
        pre_norm_bias = message.pop("pre norm bias")
    if md.add_bias_linear:
        l1_bias = message.pop("mlp l1 bias")
        if md.swiglu:
            l0_bias_W = torch.chunk(message.pop("mlp l0 bias W"), tp_size, dim=0)
            l0_bias_V = torch.chunk(message.pop("mlp l0 bias V"), tp_size, dim=0)
            l0_bias = [torch.cat(bias, dim=0) for bias in zip(l0_bias_W, l0_bias_V)]
        else:
            l0_bias = torch.chunk(message.pop("mlp l0 bias"), tp_size, dim=0)

    # set data to transformer layer for mlp
    for tp_rank, model in enumerate(models):
        tf_layer = model.decoder.layers[layer_id]
        tf_layer.mlp.linear_fc1.weight.data.copy_(l0_weight[tp_rank])
        tf_layer.mlp.linear_fc2.weight.data.copy_(l1_weight[tp_rank])
        tf_layer.mlp.linear_fc1.layer_norm_weight.data.copy_(pre_norm_weight)
        if md.norm_has_bias:
            tf_layer.mlp.linear_fc1.layer_norm_bias.data.copy(pre_norm_bias)
        if md.add_bias_linear:
            tf_layer.mlp.linear_fc1.bias.data.copy_(l0_bias[tp_rank])
            tf_layer.mlp.linear_fc2.bias.data.copy_(l1_bias)
