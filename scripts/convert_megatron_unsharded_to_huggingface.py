import argparse
import datetime
import os
import torch
import einops
import gc
import json
import shutil
import warnings

from transformers import LlamaConfig, LlamaForCausalLM

def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)

def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)

def convert_qkv(Wqkv, num_key_value_heads, n_heads, dim, dim_per_head, rotary_interleaved_patch=False):
    # Megatron stores Wqkv as ((nheads 3 headdim), hidden_dim)
    # while we store Wqkv as ((3 nheads headdim), hidden_dim)
    hidden_dim = dim
    headdim = dim_per_head
    nheads = n_heads

    # GQA compatible
    num_query_groups = num_key_value_heads
    tmp =  nheads // num_query_groups
    new_tensor_shape = (num_query_groups,
                        tmp + 2,
                        headdim,
                        Wqkv.size()[-1])
    Wq = Wqkv.view(new_tensor_shape)[:, 0:tmp        :, :]
    Wk = Wqkv.view(new_tensor_shape)[:, tmp:tmp+1    :, :]
    Wv = Wqkv.view(new_tensor_shape)[:, tmp+1:tmp+2, :, :]
    if rotary_interleaved_patch:
        def permute(w):
            return w.view(n_heads, 1, 2, headdim // 2, hidden_dim).transpose(3, 2)#.reshape(hidden_dim*3, hidden_dim)
        Wv = permute(Wv)

    Wq = Wq.reshape(nheads * headdim, hidden_dim)
    Wk = Wk.reshape(num_key_value_heads * headdim, hidden_dim)
    Wv = Wv.reshape(num_key_value_heads * headdim, hidden_dim)

    return Wq, Wk, Wv


def convert_fc1(Wfc1):
    split_size = Wfc1.size()[0]//2
    W1, W3 = torch.split(Wfc1, split_size)
    return W1, W3

def convert_checkpoint(args):
    input_base_path = args.input_dir
    model_path = args.output_dir
    safe_serialization = args.safe_serialization

    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    n_layers = args.num_layers
    n_heads = args.num_attention_heads
    dim = args.hidden_size
    dim_per_head = dim // n_heads
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dim_per_head, 2).float() / dim_per_head))

    if args.group_query_attention:
        num_key_value_heads = args.num_query_groups  # for GQA / MQA
        key_value_dim = dim // num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        key_value_dim = dim

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # Load unsharded weights
    loaded = torch.load(os.path.join(input_base_path, "model_optim_rng.pt"), map_location="cpu")

    # Megatron weights
    model_state_dict = loaded['model']
    language_model_state_dict = model_state_dict['language_model']
    embedding_state_dict = language_model_state_dict['embedding']
    encoder_state_dict = language_model_state_dict['encoder']
    src_norm1_name = "input_layernorm.weight"
    src_qkv_name = "self_attention.query_key_value.weight"
    src_proj_name = "self_attention.dense.weight"
    src_norm2_name = "post_attention_layernorm.weight"
    src_fc1_name = "mlp.dense_h_to_4h.weight"
    src_fc2_name = "mlp.dense_4h_to_h.weight"

    # Data type
    params_dtype = torch.float
    if args.data_type == 'bf16':
        params_dtype = torch.bfloat16
    elif args.data_type == 'fp16':
        params_dtype = torch.half

    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"

        src_prefix = 'layers.' + str(layer_i) + "."
        Wq, Wk, Wv = convert_qkv(encoder_state_dict[src_prefix + src_qkv_name],
                                 num_key_value_heads, n_heads, dim, dim_per_head, args.rotary_interleaved_patch)
        Wo = encoder_state_dict[src_prefix + src_proj_name]
        W1, W3 = convert_fc1(encoder_state_dict[src_prefix + src_fc1_name])
        W2 = encoder_state_dict[src_prefix + src_fc2_name]
        norm1 = encoder_state_dict[src_prefix + src_norm1_name]
        norm2 = encoder_state_dict[src_prefix + src_norm2_name]
        del encoder_state_dict[src_prefix + src_qkv_name]
        del encoder_state_dict[src_prefix + src_proj_name]
        del encoder_state_dict[src_prefix + src_fc1_name]
        del encoder_state_dict[src_prefix + src_fc2_name]
        del encoder_state_dict[src_prefix + src_norm1_name]
        del encoder_state_dict[src_prefix + src_norm2_name]

        state_dict = {
            f"model.layers.{layer_i}.self_attn.q_proj.weight": Wq.to(params_dtype),
            f"model.layers.{layer_i}.self_attn.k_proj.weight": Wk.to(params_dtype),
            f"model.layers.{layer_i}.self_attn.v_proj.weight": Wv.to(params_dtype),
            f"model.layers.{layer_i}.self_attn.o_proj.weight": Wo.to(params_dtype),
            f"model.layers.{layer_i}.mlp.gate_proj.weight": W1.to(params_dtype),
            f"model.layers.{layer_i}.mlp.down_proj.weight": W2.to(params_dtype),
            f"model.layers.{layer_i}.mlp.up_proj.weight": W3.to(params_dtype),
            f"model.layers.{layer_i}.input_layernorm.weight": norm1.to(params_dtype),
            f"model.layers.{layer_i}.post_attention_layernorm.weight": norm2.to(params_dtype),
        }

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))
        print(f"[INFO] Layer {layer_i} is converted.")

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    word_embeddings = embedding_state_dict['word_embeddings']['weight'][:args.true_vocab_size,:]
    ln_f = encoder_state_dict['final_layernorm.weight']
    lm_head = language_model_state_dict['output_layer']['weight'][:args.true_vocab_size,:]
    state_dict = {
        "model.embed_tokens.weight": word_embeddings.to(params_dtype),
        "model.norm.weight": ln_f.to(params_dtype),
        "lm_head.weight": lm_head.to(params_dtype),
    }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))
    print(f"Num of parameters is {param_count}.")
    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    ffn_dim_multiplier = args.hidden_dim_multiplier
    multiple_of = args.multiple_of
    config = LlamaConfig(
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(dim, ffn_dim_multiplier, multiple_of),
        num_attention_heads=n_heads,
        num_hidden_layers=n_layers,
        rms_norm_eps=args.layernorm_epsilon,
        num_key_value_heads=num_key_value_heads,
        vocab_size=args.true_vocab_size,
        max_position_embeddings=args.seq_length,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    print("Loading the checkpoint in a Llama model.")
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, low_cpu_mem_usage=True, torch_dtype=params_dtype)
    # Avoid saving this as part of the config.
    del model.config._name_or_path

    print("Saving in the Transformers format.")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    shutil.rmtree(tmp_model_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", "-input_dir", "-i",
        help="folder name of input files", required=True
    )
    parser.add_argument(
        "--output-dir", "-output_dir", "-o",
        help="folder name of output files", required=True
    )
    parser.add_argument(
        '--seq-length', type=int, default=4096,
        help='Maximum sequence length to process.'
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="The number of transformer layers"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        help="The number of hidden size"
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        help="The number of attention heads"
    )
    parser.add_argument(
        '--group-query-attention',
        action='store_true',
        help='Use group-query attention.'
    )
    parser.add_argument(
        '--num-query-groups',
        type=int,
        default=8
    )
    parser.add_argument(
        "--data-type", "-data_type", "-d",
        choices=["bf16", "fp32", "fp16"],
        default="fp32", help=" data type of the parameters"
    )
    parser.add_argument(
        '--multiple-of', type=int, default=None,
        help='Multiplier for setting Feed-Forward Network hidden size when swiglu.'
    )
    parser.add_argument(
        '--hidden-dim-multiplier', type=float, default=None,
        help='Custom Multiplier for setting Feed-Forward Network hidden dim when swiglu.'
    )
    parser.add_argument(
        '--layernorm-epsilon', type=float, default=1e-5,
        help='Layer norm epsilon.'
    )
    parser.add_argument(
        '--true-vocab-size', type=int, default=100008,
        help='original size of vocab, if specified will trim padding from embedding table.'
    )
    parser.add_argument(
        '--bos-token-id', type=int, default=100006,
        help='bos-token-id.'
    )
    parser.add_argument(
        '--eos-token-id', type=int, default=100006,
        help='eos-token-id.'
    )
    parser.add_argument(
        "--safe-serialization", type=bool,
        help="Whether or not to save using `safetensors`."
    )
    parser.add_argument(
        '--rotary-interleaved-patch', action='store_true',
        help='Patch for loading models using interleaved rotary position embeddings.'
    )


    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    start_time = datetime.datetime.now()
    convert_checkpoint(args)
    run_time = datetime.datetime.now() - start_time
    print(f"[INFO] Spent {run_time} (h:m:s) to convert the model")


if __name__ == "__main__":
    main()