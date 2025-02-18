import os
import json


def load_args_hf2mg(args):

    # Read deepseek_v3 args.
    deepseek_v3_args_path = os.path.join(args.load, "config.json")
    with open(deepseek_v3_args_path) as f:
        deepseek_v3_args = json.load(f)

    # Update Megatron args.
    args.vocab_size = deepseek_v3_args["vocab_size"]
    args.hidden_size = deepseek_v3_args["hidden_size"]
    args.ffn_hidden_size = deepseek_v3_args["intermediate_size"]
    moe_intermediate_size = deepseek_v3_args["intermediate_size"]
    assert moe_intermediate_size == args.ffn_hidden_size, "mlp intermediate size is not matched with moe"
    args.num_layers = deepseek_v3_args["num_hidden_layers"]
    args.use_mtp_predictor = True
    args.num_mtp_predictor = deepseek_v3_args["num_nextn_predict_layers"]
    args.num_attention_heads = deepseek_v3_args["num_attention_heads"]
    n_shared_experts = deepseek_v3_args["n_shared_experts"]
    if n_shared_experts > 0:
        args.moe_shared_expert_intermediate_size = moe_intermediate_size
    args.num_experts = deepseek_v3_args["n_routed_experts"]
    routed_scaling_factor = deepseek_v3_args["routed_scaling_factor"]
    topk_method = deepseek_v3_args["topk_method"]
    n_group = deepseek_v3_args["n_group"]
    topk_group = deepseek_v3_args["topk_group"]
    args.moe_router_topk = deepseek_v3_args["num_experts_per_tok"]
    args.moe_layer_freq = deepseek_v3_args["moe_layer_freq"]
    # if set first k dense replace, then updating moe_layer_freq
    first_k_dense_replace = deepseek_v3_args["first_k_dense_replace"]    
    args.moe_layer_freq = eval("[0]*" + str(first_k_dense_replace) + "+[1]*" + str(args.num_layers-first_k_dense_replace))
    
    norm_topk_prob = deepseek_v3_args["norm_topk_prob"]
    args.moe_router_score_function = deepseek_v3_args["scoring_func"]
    aux_loss_alpha = deepseek_v3_args["aux_loss_alpha"]
    seq_aux = deepseek_v3_args["seq_aux"]
    args.num_query_groups = deepseek_v3_args["num_key_value_heads"]
    hidden_act = deepseek_v3_args["hidden_act"]
    args.max_position_embeddings = deepseek_v3_args["max_position_embeddings"]
    args.init_method_std = deepseek_v3_args["initializer_range"]
    args.norm_epsilon = deepseek_v3_args["rms_norm_eps"]
    use_cache = deepseek_v3_args["use_cache"]
    # pad_token_id = deepseek_v3_args["pad_token_id"]
    bos_token_id = deepseek_v3_args["bos_token_id"]
    eos_token_id = deepseek_v3_args["eos_token_id"]
    pretraining_tp = deepseek_v3_args["pretraining_tp"]
    args.untie_embeddings_and_output_weights = not deepseek_v3_args["tie_word_embeddings"]
    args.rotary_base = deepseek_v3_args["rope_theta"]
    rope_scaling = deepseek_v3_args["rope_scaling"]
    attention_bias = deepseek_v3_args["attention_bias"]
    args.attention_dropout = deepseek_v3_args["attention_dropout"]
    args.q_lora_rank = deepseek_v3_args["q_lora_rank"]
    args.kv_lora_rank = deepseek_v3_args["kv_lora_rank"]
    args.qk_head_dim = deepseek_v3_args["qk_nope_head_dim"]
    args.qk_pos_emb_head_dim = deepseek_v3_args["qk_rope_head_dim"]
    args.v_head_dim = deepseek_v3_args["v_head_dim"]
    args.multi_latent_attention = True
    args.qk_layernorm = True
    args.swiglu = True
    args.bf16 = deepseek_v3_args["torch_dtype"] == "bfloat16"
    args.fp16 = deepseek_v3_args["torch_dtype"] == "float16"
    args.moe_router_enable_expert_bias = True
    args.moe_router_bias_update_rate = 0.001
    
    args.seq_length = 4096
    args.global_batch_size = 800
    args.iteration = 1 # '0', 'release' don't work
    args.add_position_embedding = False
    args.group_query_attention = True
    args.normalization = "RMSNorm"
    args.use_rotary_position_embeddings = True
    args.moe_router_load_balancing_type = "aux_loss"
    args.add_bias_linear = False
    args.add_qkv_bias = False
    args.make_vocab_size_divisible_by = 64
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0
    args.norm_has_bias = False
    args.tokenizer_type = "Emu3TokenizerFS"
    
    return args, args

