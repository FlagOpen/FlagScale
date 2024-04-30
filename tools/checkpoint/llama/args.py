import os
import json


def load_args_hf2mg(args):

    # Read llama args.
    llama_args_path = os.path.join(args.load, "config.json")
    with open(llama_args_path) as f:
        llama_args = json.load(f)
    
    # Update Megatron args.
    args.attention_dropout = llama_args["attention_dropout"]
    args.vocab_size = llama_args["vocab_size"]
    args.hidden_dropout = llama_args["attention_dropout"]
    args.hidden_size = llama_args["hidden_size"]
    args.add_qkv_bias = llama_args.get("attention_bias", True)
    args.swiglu = llama_args["hidden_act"] == "silu"
    args.init_method_std = llama_args["initializer_range"]
    args.ffn_hidden_size = llama_args["intermediate_size"]
    args.max_position_embeddings = llama_args["max_position_embeddings"]
    args.model_type = llama_args["model_type"]
    args.num_attention_heads = llama_args["num_attention_heads"]
    args.num_layers = llama_args["num_hidden_layers"]
    args.num_query_groups = llama_args["num_key_value_heads"]
    args.norm_epsilon = llama_args["rms_norm_eps"]
    args.rotary_base = llama_args["rope_theta"]
    args.untie_embeddings_and_output_weights = not llama_args["tie_word_embeddings"]
    
    args.add_position_embedding = False
    args.encoder_seq_length = 4096
    args.iteration = 1
    args.norm_has_bias = False
    args.add_bias_linear = False
    args.position_embedding_type = "rope"
    args.group_query_attention = True
    args.normalization = "RMSNorm"


def save_args_mg2hf(args):
    from transformers import LlamaConfig
    config = LlamaConfig(
        vocab_size = args.vocab_size,
        hidden_size = args.hidden_size,
        intermediate_size = args.ffn_hidden_size,
        num_hidden_layers = args.encoder_num_layers,
        num_attention_heads = args.num_attention_heads,
        num_key_value_heads = args.num_query_groups,
        max_position_embeddings = args.max_position_embeddings,
        rms_norm_eps = args.norm_epsilon,
        tie_word_embeddings = not args.untie_embeddings_and_output_weights,
        rope_theta = args.rotary_base,
        attention_bias = args.add_qkv_bias,
        attention_dropout = args.attention_dropout,
        initializer_range = args.init_method_std,
    )
    config.save_pretrained(args.save)

    return config
