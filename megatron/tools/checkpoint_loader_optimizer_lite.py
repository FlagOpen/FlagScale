"""
This script is based on Megatron's checkpoint_loader_megatron.py, but it only load the distributed optimizer state.
"""
import json
import os
import sys
import types

import torch
from checkpoint_util_lite import (
    get_num_layers_from_args,
    get_optimizer_ckpt_paths,
    get_param_index_map_paths,
    unflatten_optimizer_ckpt,
    split_optimizer_ckpt,
    remove_optimizer_tmp,
)


def add_arguments(parser):
    group = parser.add_argument_group(title="Megatron loader")

    group.add_argument(
        "--true-vocab-size",
        type=int,
        default=None,
        help="original size of vocab, if specified will trim padding from embedding table.",
    )
    group.add_argument(
        "--vocab-file",
        type=str,
        default=None,
        help="Path to the vocab file. If specified will use this to get vocab size and "
        "trim padding from the embedding table.",
    )
    group.add_argument(
        "--megatron-path",
        type=str,
        default=None,
        help="Base directory of deepspeed repository",
    )


def _load_checkpoint(queue, args):
    # Search in directory above this
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    )
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.arguments import parse_args, validate_args, _print_args
        from megatron.global_vars import set_args, set_global_variables
        from megatron.checkpointing import load_args_from_checkpoint, load_checkpoint
        from megatron.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron import fused_kernels
    except ModuleNotFoundError:
        print(
            "Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting."
        )
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us
    sys.argv = [
        "script.py",
        "--no-masked-softmax-fusion",
        "--no-bias-gelu-fusion",
        "--no-bias-dropout-fusion",
        "--no-async-tensor-model-parallel-allreduce",
        "--use-cpu-initialization",
        "--micro-batch-size",
        "1",
        "--no-load-optim",
        "--no-load-rng",
        "--no-save-optim",
        "--no-save-rng",
        "--no-initialization",
        "--load",
        args.load_dir,
    ]

    margs = parse_args()
    margs, checkpoint_args = load_args_from_checkpoint(margs)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    margs.world_size = (
        margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size
    )

    margs = validate_args(margs)

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                print(f"Arguments: {margs}")
                queue.put("exit")
                exit(1)

    check_for_arg("tensor_model_parallel_size")
    check_for_arg("pipeline_model_parallel_size")
    check_for_arg("num_layers")
    check_for_arg("hidden_size")
    check_for_arg("seq_length")
    check_for_arg("num_attention_heads")
    check_for_arg("max_position_embeddings")
    check_for_arg("position_embedding_type")
    check_for_arg("tokenizer_type")
    check_for_arg("iteration")
    check_for_arg("bert_binary_head")
    check_for_arg("disable_bias_linear", False)
    check_for_arg("params_dtype")
    check_for_arg("swiglu", False)

    # Determine how to make our models
    if args.model_type == "GPT":
        # from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f"unrecognized model type: {args.model_type}")

    set_global_variables(margs, build_tokenizer=False)

    # Get true (non-padded) vocab size
    if args.true_vocab_size is not None:
        true_vocab_size = args.true_vocab_size
    elif args.vocab_file is not None:
        vocab = json.load(open(args.vocab_file))
        true_vocab_size = len(vocab)
        if args.true_vocab_size is not None and true_vocab_size != args.true_vocab_size:
            print(
                "Both --true-vocab-size and --vocab-file specified and the vocab size does not match, aborting."
            )
            queue.put("exit")
            exit(1)
    else:
        true_vocab_size = None

    # short aliases
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    hetero_pipeline_stages = checkpoint_args.hetero_pipeline_stages
    hetero_pipeline_stage_splits = checkpoint_args.hetero_pipeline_stage_splits
    if vp_size is None:
        vp_size = 1

    # norm has bias; RMSNorm does not.
    if hasattr(checkpoint_args, 'normalization'):
        norm_has_bias = checkpoint_args.normalization == "norm"
    else:
        # older models only supported norm
        norm_has_bias = True

    # metadata
    md = types.SimpleNamespace()
    md.load = margs.load
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = norm_has_bias
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = true_vocab_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = checkpoint_args

    if not args.loader_mapping_from_start:
        param_index_map_paths = get_param_index_map_paths(
            md.load, tp_size, pp_size, md.iteration
        )
    else:
        param_index_map_paths = get_param_index_map_paths(
             md.load, tp_size, pp_size, 0 
        )
    param_index_maps = [[None for _ in range(pp_size)] for _ in range(tp_size)]
    def get_param_index_map(param_index_maps, param_index_map_paths, tp_rank, pp_rank):
        if param_index_maps[tp_rank][pp_rank] is None:
            param_index_map_path = param_index_map_paths[tp_rank][pp_rank]
            with open(param_index_map_path, "r") as f:
                param_index_maps[tp_rank][pp_rank] = json.load(f)
            return param_index_maps[tp_rank][pp_rank]
        else:
            return param_index_maps[tp_rank][pp_rank]

    optimizer_ckpt_paths = get_optimizer_ckpt_paths(
        md.load, tp_size, pp_size, md.iteration
    )

    def get_optimizer_ckpt(
        optimizer_ckpts,
        optimizer_ckpt_paths,
        state_key,
        tp_rank,
        pp_rank,
        vp_size,
    ):
        assert state_key in ["param", "exp_avg", "exp_avg_sq"]
        if optimizer_ckpts[tp_rank][pp_rank] is None:
            optimizer_ckpt_path = optimizer_ckpt_paths[tp_rank][pp_rank]
            ckpt_name, ckpt_ext = os.path.splitext(optimizer_ckpt_path)
            load_path = ckpt_name + f"_{state_key}" + ckpt_ext
            print(f"Loading optimizer state from {load_path}")
            optimizer_ckpt = torch.load(load_path, map_location="cpu")
            param_index_map = get_param_index_map(
                param_index_maps, param_index_map_paths, tp_rank, pp_rank
            )
            unflatten_ckpt = unflatten_optimizer_ckpt(
                optimizer_ckpt, param_index_map, vp_size, state_key
            )
            optimizer_ckpts[tp_rank][pp_rank] = unflatten_ckpt
            return optimizer_ckpts[tp_rank][pp_rank]
        else:
            return optimizer_ckpts[tp_rank][pp_rank]

    def get_optimizer_state(
        optimizer_ckpt,
        layer_num,
        vp_rank,
        state_key,
        param_key,
        bias=False,
        dtype=torch.float32,
    ):
        assert state_key in ["param", "exp_avg", "exp_avg_sq"]
        old_full_key = None
        if param_key == "word_embeddings":
            full_key = "module.language_model.embedding.word_embeddings.weight"
        elif param_key == "input_norm":
            if not bias:
                full_key = f"module.language_model.encoder.layers.{layer_num}.input_norm.weight"
                old_full_key = f"module.language_model.encoder.layers.{layer_num}.input_layernorm.weight"
            else:
                full_key = f"module.language_model.encoder.layers.{layer_num}.input_norm.bias"
                old_full_key = f"module.language_model.encoder.layers.{layer_num}.input_layernorm.bias"
        elif param_key == "query_key_value":
            if not bias:
                full_key = f"module.language_model.encoder.layers.{layer_num}.self_attention.query_key_value.weight"
            else:
                full_key = f"module.language_model.encoder.layers.{layer_num}.self_attention.query_key_value.bias"
        elif param_key == "dense":
            if not bias:
                full_key = f"module.language_model.encoder.layers.{layer_num}.self_attention.dense.weight"
            else:
                full_key = f"module.language_model.encoder.layers.{layer_num}.self_attention.dense.bias"
        elif param_key == "post_attention_norm":
            if not bias:
                full_key = f"module.language_model.encoder.layers.{layer_num}.post_attention_norm.weight"
                old_full_key = f"module.language_model.encoder.layers.{layer_num}.post_attention_layernorm.weight"
            else:
                full_key = f"module.language_model.encoder.layers.{layer_num}.post_attention_norm.bias"
                old_full_key = f"module.language_model.encoder.layers.{layer_num}.post_attention_layernorm.bias"
        elif param_key == "dense_h_to_4h":
            if not bias:
                full_key = f"module.language_model.encoder.layers.{layer_num}.mlp.dense_h_to_4h.weight"
            else:
                full_key = f"module.language_model.encoder.layers.{layer_num}.mlp.dense_h_to_4h.bias"
        elif param_key == "dense_4h_to_h":
            if not bias:
                full_key = f"module.language_model.encoder.layers.{layer_num}.mlp.dense_4h_to_h.weight"
            else:
                full_key = f"module.language_model.encoder.layers.{layer_num}.mlp.dense_4h_to_h.bias"
        elif param_key == "final_norm":
            if not bias:
                full_key = f"module.language_model.encoder.final_norm.weight"
                old_full_key = f"module.language_model.encoder.final_layernorm.weight"
            else:
                full_key = f"module.language_model.encoder.final_norm.bias"
                old_full_key = f"module.language_model.encoder.final_layernorm.bias"
        elif param_key == "output_layer":
            full_key = f"module.language_model.output_layer.weight"
        else:
            print(
                "[WARNING]: loading unrecognized state key: "
                + state_key
                + ", param_key "
                + param_key
            )
            exit(1)
        if full_key in optimizer_ckpt[vp_rank][dtype][state_key]:
            return optimizer_ckpt[vp_rank][dtype][state_key][full_key]
        elif old_full_key and \
            old_full_key in optimizer_ckpt[vp_rank][dtype][state_key]:
            return optimizer_ckpt[vp_rank][dtype][state_key][old_full_key]
        else:
            raise Exception(f"key {full_key} or {old_full_key} not found in optimizer checkpoint")

    # ------- split optimizer ckpts -------
    for tp_rank in range(tp_size):
        for pp_rank in range(pp_size):
            split_optimizer_ckpt(optimizer_ckpt_paths[tp_rank][pp_rank], vp_size)

    # ------- send optimizer ckpts -------
    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    def send_message(state_key):
        optimizer_ckpts = [[None for _ in range(pp_size)] for _ in range(tp_size)]
        # Send embeddings
        assert md.position_embedding_type != "learned_absolute"
        message = {
            f"word embeddings {state_key}": torch.cat(
                [
                    get_optimizer_state(
                        get_optimizer_ckpt(
                            optimizer_ckpts,
                            optimizer_ckpt_paths,
                            state_key,
                            tp_rank,
                            0,
                            vp_size,
                        ),
                        None,
                        0,
                        state_key,
                        "word_embeddings",
                    )
                    for tp_rank in range(tp_size)
                ],
                dim=0,
            ),
        }
        queue_put(f"embeddings {state_key}", message)

        total_layer_num = 0
        for vp_rank in range(vp_size):
            for pp_rank in range(pp_size):
                num_layers = get_num_layers_from_args(md.num_layers, pp_size, pp_rank,
                                                      hetero_pipeline_stages, hetero_pipeline_stage_splits)
                for layer_num in range(num_layers):
                    message = {}
                    # Get non-parallel tensors from tp_rank 0
                    optimizer_ckpt = get_optimizer_ckpt(
                        optimizer_ckpts,
                        optimizer_ckpt_paths,
                        state_key,
                        0,
                        pp_rank,
                        vp_size,
                    )
                    message[
                        f"input norm weight {state_key}"
                    ] = get_optimizer_state(
                        optimizer_ckpt, layer_num, vp_rank, state_key, "input_norm"
                    )
                    if md.norm_has_bias:
                        message[
                            f"input norm bias {state_key}"
                        ] = get_optimizer_state(
                            optimizer_ckpt,
                            layer_num,
                            vp_rank,
                            state_key,
                            "input_norm",
                            True,
                        )
                    message[f"post norm weight {state_key}"] = get_optimizer_state(
                        optimizer_ckpt,
                        layer_num,
                        vp_rank,
                        state_key,
                        "post_attention_norm",
                    )
                    if md.norm_has_bias:
                        message[
                            f"post norm bias {state_key}"
                        ] = get_optimizer_state(
                            optimizer_ckpt,
                            layer_num,
                            vp_rank,
                            state_key,
                            "post_attention_norm",
                            True,
                        )
                    if md.linear_bias:
                        message[f"dense bias {state_key}"] = get_optimizer_state(
                            optimizer_ckpt, layer_num, vp_rank, state_key, "dense", True
                        )
                        message[f"mlp l1 bias {state_key}"] = get_optimizer_state(
                            optimizer_ckpt,
                            layer_num,
                            vp_rank,
                            state_key,
                            "dense_4h_to_h",
                            True,
                        )
                    qkv_weight = []
                    qkv_bias = []
                    dense_weight = []
                    mlp_l0_weight = []
                    mlp_l0_bias = []
                    mlp_l1_weight = []
                    for tp_rank in range(tp_size):
                        optimizer_ckpt = get_optimizer_ckpt(
                            optimizer_ckpts,
                            optimizer_ckpt_paths,
                            state_key,
                            tp_rank,
                            pp_rank,
                            vp_size,
                        )
                        qkv_weight.append(
                            get_optimizer_state(
                                optimizer_ckpt,
                                layer_num,
                                vp_rank,
                                state_key,
                                "query_key_value",
                            )
                        )
                        dense_weight.append(
                            get_optimizer_state(
                                optimizer_ckpt, layer_num, vp_rank, state_key, "dense"
                            )
                        )
                        mlp_l0_weight.append(
                            get_optimizer_state(
                                optimizer_ckpt,
                                layer_num,
                                vp_rank,
                                state_key,
                                "dense_h_to_4h",
                            )
                        )
                        mlp_l1_weight.append(
                            get_optimizer_state(
                                optimizer_ckpt,
                                layer_num,
                                vp_rank,
                                state_key,
                                "dense_4h_to_h",
                            )
                        )
                        if md.linear_bias:
                            qkv_bias.append(
                                get_optimizer_state(
                                    optimizer_ckpt,
                                    layer_num,
                                    vp_rank,
                                    state_key,
                                    "query_key_value",
                                    True,
                                )
                            )
                            mlp_l0_bias.append(
                                get_optimizer_state(
                                    optimizer_ckpt,
                                    layer_num,
                                    vp_rank,
                                    state_key,
                                    "dense_h_to_4h",
                                    True,
                                )
                            )

                    # Handle gated linear units
                    if md.swiglu:
                        # concat all the first halves ('W's) and all the second halves ('V's)
                        for tp_rank in range(tp_size):
                            mlp_l0_weight[tp_rank] = torch.chunk(
                                mlp_l0_weight[tp_rank], 2, dim=0
                            )
                        message[f"mlp l0 weight W {state_key}"] = torch.cat(
                            [w[0] for w in mlp_l0_weight], dim=0
                        )
                        message[f"mlp l0 weight V {state_key}"] = torch.cat(
                            [w[1] for w in mlp_l0_weight], dim=0
                        )
                    else:
                        message[f"mlp l0 weight {state_key}"] = torch.cat(
                            mlp_l0_weight, dim=0
                        )

                    # simple concat of the rest
                    message[f"qkv weight {state_key}"] = torch.cat(qkv_weight, dim=0)
                    message[f"dense weight {state_key}"] = torch.cat(
                        dense_weight, dim=1
                    )
                    message[f"mlp l1 weight {state_key}"] = torch.cat(
                        mlp_l1_weight, dim=1
                    )
                    if md.linear_bias:
                        message[f"qkv bias {state_key}"] = torch.cat(qkv_bias, dim=0)
                        if md.swiglu:
                            for tp_rank in range(tp_size):
                                mlp_l0_bias[tp_rank] = torch.chunk(
                                    mlp_l0_bias[tp_rank], 2, dim=0
                                )
                            message[f"mlp l0 bias W {state_key}"] = torch.cat(
                                [b[0] for b in mlp_l0_bias], dim=0
                            )
                            message[f"mlp l0 bias V {state_key}"] = torch.cat(
                                [b[1] for b in mlp_l0_bias], dim=0
                            )
                        else:
                            message[f"mlp l0 bias {state_key}"] = torch.cat(
                                mlp_l0_bias, dim=0
                            )

                    queue_put(
                        f"transformer layer {total_layer_num} {state_key}", message
                    )

                    total_layer_num = total_layer_num + 1

        # Send final norm main weight from tp_rank 0
        optimizer_ckpt = get_optimizer_ckpt(
            optimizer_ckpts,
            optimizer_ckpt_paths,
            state_key,
            0,
            pp_rank,
            vp_size,
        )
        if md.norm_has_bias:
            message = {
                f"weight {state_key}": get_optimizer_state(
                    optimizer_ckpt, None, vp_rank, state_key, "final_norm"
                ),
                f"bias {state_key}": get_optimizer_state(
                    optimizer_ckpt, vp_rank, state_key, "final_norm", True
                ),
            }
        else:
            message = {
                f"weight {state_key}": get_optimizer_state(
                    optimizer_ckpt, None, vp_rank, state_key, "final_norm"
                ),
            }
        queue_put(f"final norm {state_key}", message)

        if md.output_layer:
            # Send output_layer main weight tp_rank 0
            message = {
                f"weight {state_key}": torch.cat(
                    [
                        get_optimizer_state(
                            get_optimizer_ckpt(
                                optimizer_ckpts,
                                optimizer_ckpt_paths,
                                state_key,
                                tp_rank,
                                pp_rank,
                                vp_size,
                            ),
                            None,
                            vp_rank,
                            state_key,
                            "output_layer",
                        )
                        for tp_rank in range(tp_size)
                    ],
                    dim=0,
                )
            }
            queue_put(f"output layer {state_key}", message)

    send_message("param")

    send_message("exp_avg")

    send_message("exp_avg_sq")

    queue.put("done")

    # ------- remove the splitted optimizer ckpts -------
    if not args.no_del_tmp:
        for tp_rank in range(tp_size):
            for pp_rank in range(pp_size):
                remove_optimizer_tmp(optimizer_ckpt_paths[tp_rank][pp_rank])


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
