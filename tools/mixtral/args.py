import os
import json


def load_args_hf2mg(args):

    # Read mixtral args.
    mixtral_args_path = os.path.join(args.load, "config.json")
    with open(mixtral_args_path) as f:
        mixtral_args = json.load(f)

    # Update Megatron args.
    args.seq_length = 2048
    args.attention_dropout = mixtral_args["attention_dropout"]
    args.hidden_dropout = mixtral_args["attention_dropout"]
    args.swiglu = True
    args.hidden_size = mixtral_args["hidden_size"]
    args.init_method_std = mixtral_args["initializer_range"]
    args.ffn_hidden_size = mixtral_args["intermediate_size"]
    args.max_position_embeddings = mixtral_args["max_position_embeddings"]
    args.model_type = mixtral_args["model_type"]
    args.num_attention_heads = mixtral_args["num_attention_heads"]
    args.moe_router_topk = mixtral_args["num_experts_per_tok"]
    args.num_layers = mixtral_args["num_hidden_layers"]
    args.group_query_attention = True
    args.num_query_groups = mixtral_args["num_key_value_heads"]
    args.num_experts = mixtral_args["num_local_experts"]
    args.normalization = "RMSNorm"
    args.norm_epsilon = mixtral_args["rms_norm_eps"]
    args.use_rotary_position_embeddings = True
    args.moe_router_load_balancing_type = "aux_loss"
    args.moe_aux_loss_coeff = mixtral_args["router_aux_loss_coef"]
    args.untie_embeddings_and_output_weights = not mixtral_args["tie_word_embeddings"]
    args.bf16 = mixtral_args["torch_dtype"] == "bfloat16"
    args.fp16 = mixtral_args["torch_dtype"] == "float16"
    args.vocab_size = mixtral_args["vocab_size"]
    args.padded_vocab_size = mixtral_args["vocab_size"]

    args.global_batch_size = 1024
    args.iteration = 1 # '0', 'release' don't work
    args.add_position_embedding = False
    args.add_bias_linear = False
    args.add_qkv_bias = False
    args.norm_has_bias = False
    args.true_vocab_size = None # to skip padding in saver
    args.make_vocab_size_divisible_by = None
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0

    return args, args


