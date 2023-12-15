import argparse
import datetime
import os
import torch
import einops
import gc
import json
import shutil
import warnings
from copy import deepcopy

from transformers import LlamaConfig, LlamaForCausalLM

def reverse_qkv(Wq, Wk, Wv, num_key_value_heads, n_heads, headdim):
    # number of q heads in each GQA group
    tmp = int(n_heads / num_key_value_heads)
    # print(num_key_value_heads, tmp, headdim, Wq.size(-1))
    Wq = Wq.view(num_key_value_heads, tmp, headdim, Wq.size(-1))
    Wk = Wk.view(num_key_value_heads, 1, headdim, Wk.size(-1))
    Wv = Wv.view(num_key_value_heads, 1, headdim, Wv.size(-1))

    # Stack Wq, Wk, and Wv to reconstruct Wqkv
    Wqkv = torch.cat([Wq, Wk, Wv], dim=1)
    Wqkv = Wqkv.view(-1, Wq.size(-1))

    return Wqkv

def reverse_fc1(W1, W3):
    return torch.cat([W1, W3], dim=0)

def reverse_checkpoint(args):
    input_base_path = args.input_dir
    model_path = args.output_dir
    ref_model_path = args.ref_dir

    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    n_layers = args.num_layers
    n_heads = args.num_attention_heads
    dim = args.hidden_size
    dim_per_head = dim // n_heads
    base = 10000.0

    if args.group_query_attention:
        num_key_value_heads = args.num_query_groups  # for GQA / MQA
        key_value_dim = dim // num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        key_value_dim = dim

    # Data type
    params_dtype = torch.float
    if args.data_type == 'bf16':
        params_dtype = torch.bfloat16
    elif args.data_type == 'fp16':
        params_dtype = torch.half

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    model = LlamaForCausalLM.from_pretrained(input_base_path, low_cpu_mem_usage=True, torch_dtype=params_dtype)

    print(f"Load ref model at {ref_model_path}")
    ref = torch.load(os.path.join(ref_model_path, "model_optim_rng.pt"), map_location="cpu")

    # load args from another megatron ckpt
    out_model = deepcopy(ref)
    out_model['model']['language_model']['encoder'] = {}

    for layer_i in range(n_layers):
        Wqkv = reverse_qkv(model.model.layers[layer_i].self_attn.q_proj.weight, model.model.layers[layer_i].self_attn.k_proj.weight, model.model.layers[layer_i].self_attn.v_proj.weight, num_key_value_heads, n_heads, dim_per_head)
        fc1 = reverse_fc1(model.model.layers[layer_i].mlp.gate_proj.weight, model.model.layers[layer_i].mlp.up_proj.weight)
        fc2 = model.model.layers[layer_i].mlp.down_proj.weight
        norm1 = model.model.layers[layer_i].input_layernorm.weight
        norm2 = model.model.layers[layer_i].post_attention_layernorm.weight
        Wo = model.model.layers[layer_i].self_attn.o_proj.weight

        out_model['model']['language_model']['encoder'][f'layers.{layer_i}.self_attention.query_key_value.weight'] = Wqkv.to(params_dtype)
        out_model['model']['language_model']['encoder'][f'layers.{layer_i}.mlp.dense_h_to_4h.weight'] = fc1.to(params_dtype)
        out_model['model']['language_model']['encoder'][f'layers.{layer_i}.mlp.dense_4h_to_h.weight'] = fc2.to(params_dtype)
        out_model['model']['language_model']['encoder'][f'layers.{layer_i}.input_layernorm.weight'] = norm1.to(params_dtype)
        out_model['model']['language_model']['encoder'][f'layers.{layer_i}.post_attention_layernorm.weight'] = norm2.to(params_dtype)
        out_model['model']['language_model']['encoder'][f'layers.{layer_i}.self_attention.dense.weight'] = Wo.to(params_dtype)

        del Wqkv, fc1, fc2, norm1, norm2, Wo
        print(f"[INFO] Layer {layer_i} is converted.")

    if args.vocab_size > model.model.embed_tokens.weight.size(0):
        pad_size = args.vocab_size - model.model.embed_tokens.weight.size(0)
        pad_weight = torch.zeros([pad_size, model.model.embed_tokens.weight.size(1)]).to(params_dtype)
        out_model['model']['language_model']['embedding']['word_embeddings']['weight'] = torch.cat([model.model.embed_tokens.weight.to(params_dtype), pad_weight], dim=0)
        out_model['model']['language_model']['output_layer']['weight'] = torch.cat([model.lm_head.weight.to(params_dtype), pad_weight], dim=0)
    else:
        out_model['model']['language_model']['embedding']['word_embeddings']['weight'] = model.model.embed_tokens.weight[:args.vocab_size].to(params_dtype)
        out_model['model']['language_model']['output_layer']['weight'] = model.lm_head.weight[:args.vocab_size].to(params_dtype)

    out_model['model']['language_model']['encoder']['final_layernorm.weight'] = model.model.norm.weight.to(params_dtype)

    out_model['iteration'] = args.iteration

    print("Saving the converted Megatron Model.")
    formatted_str = f"{args.iteration:07d}"
    f_path =  os.path.join(model_path, "iter_" + formatted_str + "/mp_rank_00")
    os.makedirs(f_path, exist_ok=True)
    torch.save(out_model, os.path.join(f_path, "model_optim_rng.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-input_dir", "-i",
                        help="folder name of input files", required=True)
    parser.add_argument("--output-dir", "-output_dir", "-o",
                        help="folder name of output files", required=True)
    parser.add_argument("--ref-dir", "-ref_dir", "-r",
                        help="folder name of ref model files, mainly to copy the args setting", required=True)
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
        "--data-type", "-data_type", "-d", choices=["fp32", "fp16", "bf16"], default="fp32", help=" data type of the parameters"
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
        '--iteration',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--vocab-size', type=int, default=100032,
        help='size of vocab, if specified will add padding from embedding table.'
    )


    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    start_time = datetime.datetime.now()
    reverse_checkpoint(args)
    run_time = datetime.datetime.now() - start_time
    print(f"[INFO] Spent {run_time} (h:m:s) to convert the model")

if __name__ == "__main__":
    main()
