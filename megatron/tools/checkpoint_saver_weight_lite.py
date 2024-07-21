"""
This script is based on Megatron's checkpoint_saver_megatron.py, but it only saves the model ckpt without build the model.
"""
import argparse
from collections.abc import Mapping
from collections import OrderedDict
import concurrent.futures
import os
import sys
import json

import torch
from checkpoint_util_lite import (
    get_model_ckpt_paths,
    get_num_layers_from_args,
)


def add_arguments(parser):
    group = parser.add_argument_group(title="Megatron saver")

    group.add_argument(
        "--megatron-path",
        type=str,
        default=None,
        help="Base directory of Megatron repository",
    )

    group.add_argument(
        "--target-tensor-parallel-size",
        type=int,
        help="Target tensor model parallel size, defaults to the tensor parallel size "
        "in the input checkpoint if provided by the loader, otherwise to 1",
    )
    group.add_argument(
        "--target-pipeline-parallel-size",
        type=int,
        help="Target tensor model parallel size, default to the pipeline parall size "
        "in the input checkpoint if provided by the loader, otherwise to 1",
    )

def save_checkpoint(queue, args):
    # Search in directory above this
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    )
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.arguments import parse_args, validate_args, _print_args
        from megatron.checkpointing import save_checkpoint
        from megatron.global_vars import set_global_variables, get_args
        from megatron.core.enums import ModelType
        from megatron.tokenizer.tokenizer import _vocab_size_with_padding
        from megatron import fused_kernels
        from megatron.core import mpu
    except ModuleNotFoundError:
        print(
            "Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting."
        )
        exit(1)

    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(
                f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.'
            )
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(
                f"Exiting. If you want to ignore this, use the argument --no-checking."
            )
            exit(1)

    md = queue_get()

    if args.target_tensor_parallel_size is None:
        if hasattr(md, "previous_tensor_parallel_size"):
            args.target_tensor_parallel_size = md.previous_tensor_parallel_size
        else:
            print(
                "loader did not provide a tensor parallel size and --target-tensor-parallel-size not provided on command line. "
                "Default to 1."
            )
            args.target_tensor_parallel_size = 1

    if args.target_pipeline_parallel_size is None:
        if hasattr(md, "previous_pipeline_parallel_size"):
            args.target_pipeline_parallel_size = md.previous_pipeline_parallel_size
        else:
            print(
                "loader did not provide a pipeline parallel size and --target-pipeline-parallel-size not provided on command line. "
                "Default to 1."
            )
            args.target_pipeline_parallel_size = 1

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    if (
        args.target_tensor_parallel_size is not None
        and args.target_pipeline_parallel_size is not None
    ):
        os.environ[
            "WORLD_SIZE"
        ] = f"{args.target_tensor_parallel_size * args.target_pipeline_parallel_size}"
    # We want all arguments to come from us
    sys.argv = [
        "script.py",
        "--num-layers",
        str(md.num_layers),
        "--hidden-size",
        str(md.hidden_size),
        "--seq-length",
        str(md.seq_length),
        "--num-attention-heads",
        str(md.num_attention_heads),
        "--max-position-embeddings",
        str(md.max_position_embeddings),
        "--position-embedding-type",
        str(md.position_embedding_type),
        "--tokenizer-type",
        str(md.tokenizer_type),
        "--tensor-model-parallel-size",
        str(args.target_tensor_parallel_size),
        "--pipeline-model-parallel-size",
        str(args.target_pipeline_parallel_size),
        "--no-masked-softmax-fusion",
        "--no-bias-gelu-fusion",
        "--no-bias-dropout-fusion",
        "--no-async-tensor-model-parallel-allreduce",
        "--use-cpu-initialization",
        "--micro-batch-size",
        "1",
        "--no-load-rng",
        "--no-save-optim",
        "--no-save-rng",
        "--no-initialization",
        "--save-interval",
        "1",
        "--save",
        args.save_dir,
    ]

    if md.make_vocab_size_divisible_by is not None:
        sys.argv.extend(
            ["--make-vocab-size-divisible-by", str(md.make_vocab_size_divisible_by)]
        )
    if md.params_dtype == torch.float16:
        sys.argv.append("--fp16")
    elif md.params_dtype == torch.bfloat16:
        sys.argv.append("--bf16")

    if md.output_layer:
        sys.argv.append("--untie-embeddings-and-output-weights")
    if not md.linear_bias:
        sys.argv.append("--disable-bias-linear")

    margs = parse_args()

    margs.hetero_pipeline_layer_split = args.hetero_pipeline_layer_split
    if args.hetero_pipeline_layer_split is not None:
        margs.enable_hetero = True 

    if hasattr(md, "checkpoint_args"):
        # These are arguments that we are either changing, or cause problems for validation if they are set
        # Note that some of these deal with T5 so will need to be changed if we support T5.
        args_to_keep = [
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "world_size",
            "params_dtype",
            "num_layers_per_virtual_pipeline_stage",
            "virtual_pipeline_model_parallel_size",
            "masked_softmax_fusion",
            "bias_gelu_fusion",
            "bias_dropout_fusion",
            "sequence_parallel",
            "async_tensor_model_parallel_allreduce",
            "no_load_optim",
            "no_load_rng",
            "no_save_optim",
            "no_save_rng",
            "vocab_file",
            "tokenizer_model",
            "save_interval",
            "save",
            "perform_initialization",
            "use_cpu_initialization",
            "encoder_num_layers",
            "encoder_seq_length",
            "distribute_saved_activations",
            "train_iters",
            "lr_decay_iters",
            "lr_warmup_iters",
            "lr_warmup_fraction",
            "start_weight_decay",
            "end_weight_decay",
            "enable_hetero",
            "hetero_pipeline_layer_split",
        ]

        for arg, value in vars(md.checkpoint_args).items():
            if arg in args_to_keep:
                continue
            if not hasattr(margs, arg):
                print(
                    f"Checkpoint had argument {arg} but new arguments does not have this."
                )
                continue
            if getattr(margs, arg) != value:
                print(
                    f"Overwriting default {arg} value {getattr(margs, arg)} with value from checkpoint {value}."
                )
                setattr(margs, arg, value)

    validate_args(margs)

    set_global_variables(margs, build_tokenizer=False)

    # margs = megatron args
    margs = get_args()

    if hasattr(md, "consumed_train_samples"):
        margs.consumed_train_samples = md.consumed_train_samples
        margs.consumed_valid_samples = md.consumed_valid_samples
        print(
            f"Setting consumed_train_samples to {margs.consumed_train_samples}"
            f" and consumed_valid_samples to {margs.consumed_valid_samples}"
        )
    else:
        print("consumed_train_samples not provided.")

    # Determine how to make our models
    if md.model_type == "GPT":
        # from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f"unrecognized model type: {args.model_type}")

    margs.iteration = md.iteration

    tp_size = args.target_tensor_parallel_size
    pp_size = args.target_pipeline_parallel_size
    vp_size = 1
    assert margs.tensor_model_parallel_size == tp_size
    assert margs.pipeline_model_parallel_size == pp_size

    model_ckpts = [[None for _ in range(pp_size)] for _ in range(tp_size)]
    def get_model_ckpt(model_ckpts, tp_rank, pp_rank, vp_size, vp_rank):
        if vp_size > 1:
            model_key = "model" + str(vp_rank)
        else:
            model_key = "model"
        if model_ckpts[tp_rank][pp_rank] is None:
            model_ckpt = {
                "args": None,
                "checkpoint_version": None,
                "iteration": None,
                model_key: None,
                "optimizer": None,
                "opt_param_scheduler": None,
                "rng_state": None,
            }
            model_ckpt[model_key] = {"language_model": None}
            model_ckpt[model_key]["language_model"] = {
                "embedding": None,
                "encoder": None,
                "output_layer": None,
            }
            model_ckpt[model_key]["language_model"]["embedding"] = {
                "word_embeddings": None,
                "position_embeddings": None,
            }
            model_ckpt[model_key]["language_model"]["embedding"][
                "word_embeddings"
            ] = OrderedDict()
            model_ckpt[model_key]["language_model"]["encoder"] = OrderedDict()
            model_ckpt[model_key]["language_model"]["output_layer"] = OrderedDict()
            model_ckpts[tp_rank][pp_rank] = model_ckpt
            return model_ckpts[tp_rank][pp_rank]
        else:
            return model_ckpts[tp_rank][pp_rank]

    def set_weight_or_bias(
        model_ckpt, layer_num, vp_size, vp_rank, key, val, bias=False
    ):
        if vp_size > 1:
            model_key = "model" + str(vp_rank)
        else:
            model_key = "model"
        if key == "word_embeddings":
            model_ckpt[model_key]["language_model"]["embedding"]["word_embeddings"][
                "weight"
            ] = val
        elif key == "input_norm":
            if not bias:
                full_key = "layers." + str(layer_num) + "." + key + ".weight"
            else:
                full_key = "layers." + str(layer_num) + "." + key + ".bias"
            model_ckpt[model_key]["language_model"]["encoder"][full_key] = val
        elif key == "query_key_value":
            if not bias:
                full_key = (
                    "layers." + str(layer_num) + ".self_attention." + key + ".weight"
                )
            else:
                full_key = (
                    "layers." + str(layer_num) + ".self_attention." + key + ".bias"
                )
            model_ckpt[model_key]["language_model"]["encoder"][full_key] = val
        elif key == "dense":
            if not bias:
                full_key = (
                    "layers." + str(layer_num) + ".self_attention." + key + ".weight"
                )
            else:
                full_key = (
                    "layers." + str(layer_num) + ".self_attention." + key + ".bias"
                )
            model_ckpt[model_key]["language_model"]["encoder"][full_key] = val
        elif key == "post_attention_norm":
            if not bias:
                full_key = "layers." + str(layer_num) + "." + key + ".weight"
            else:
                full_key = "layers." + str(layer_num) + "." + key + ".bias"
            model_ckpt[model_key]["language_model"]["encoder"][full_key] = val
        elif key == "dense_h_to_4h":
            if not bias:
                full_key = "layers." + str(layer_num) + ".mlp." + key + ".weight"
            else:
                full_key = "layers." + str(layer_num) + ".mlp." + key + ".bias"
            model_ckpt[model_key]["language_model"]["encoder"][full_key] = val
        elif key == "dense_4h_to_h":
            if not bias:
                full_key = "layers." + str(layer_num) + ".mlp." + key + ".weight"
            else:
                full_key = "layers." + str(layer_num) + ".mlp." + key + ".bias"
            model_ckpt[model_key]["language_model"]["encoder"][full_key] = val
        elif key == "final_norm":
            if not bias:
                model_ckpt[model_key]["language_model"]["encoder"][
                    "final_norm.weight"
                ] = val
            else:
                model_ckpt[model_key]["language_model"]["encoder"][
                    "final_norm.bias"
                ] = val
        elif key == "output_layer":
            model_ckpt[model_key]["language_model"]["output_layer"]["weight"] = val
        else:
            print("[WARNING]: unrecognized key: " + key)

    def set_optimizer_info(model_ckpt, optimizer, opt_param_scheduler, args):
        model_ckpt['optimizer'] = optimizer
        model_ckpt['opt_param_scheduler'] = opt_param_scheduler
        model_ckpt['iteration'] = args.iteration
        model_ckpt['checkpoint_version'] = 3.0
        model_ckpt['args'] = args 

    model_ckpt_paths = get_model_ckpt_paths(margs.save, tp_size, pp_size, md.iteration)
    def save_model_ckpt(model_ckpts, model_ckpt_paths, tp_rank, pp_rank):
        if model_ckpts[tp_rank][pp_rank] is not None:
            model_ckpt_path = model_ckpt_paths[tp_rank][pp_rank]
            model_ckpt_dir = os.path.dirname(model_ckpt_path)
            if not os.path.exists(model_ckpt_dir):
                os.makedirs(model_ckpt_dir)
            torch.save(model_ckpts[tp_rank][pp_rank], model_ckpt_path)
            # Delete the saved model from memory
            model_ckpt = model_ckpts[tp_rank][pp_rank]
            model_ckpts[tp_rank][pp_rank] = None
            del model_ckpt
        else:
            raise Exception("model ckpt is None")

    def padding_vocab(orig_word_embed):
        # Deal with padding
        if md.true_vocab_size is not None:
            # figure out what our padded vocab size is
            orig_vocab_size = orig_word_embed.shape[0]
            margs.padded_vocab_size = _vocab_size_with_padding(md.true_vocab_size, margs)

            # Cut out extra padding we don't need
            if orig_vocab_size > margs.padded_vocab_size:
                full_word_embed = orig_word_embed[0 : margs.padded_vocab_size, :]

            # Expanding embedding to larger size by replicating final entry
            elif orig_vocab_size < margs.padded_vocab_size:
                padding_size = margs.padded_vocab_size - orig_vocab_size

                full_word_embed = torch.cat(
                    (
                        orig_word_embed,
                        orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1),
                    )
                )

            # Same size!
            else:
                full_word_embed = orig_word_embed
        else:
            print(
                "Original vocab size not specified, leaving embedding table as-is. "
                "If you've changed the tensor parallel size this could cause problems."
            )
            margs.padded_vocab_size = orig_word_embed.shape[0]
            full_word_embed = orig_word_embed
        return full_word_embed

    # Optimizer
    optimizer_info_msg = queue_get("optimizer info")
    optimizer = optimizer_info_msg["optimizer"]
    opt_param_scheduler = optimizer_info_msg["opt_param_scheduler"]

    # Embeddings
    # -----------
    embeddings_msg = queue_get("embeddings")
    assert md.position_embedding_type != "learned_absolute"
    orig_word_embed = embeddings_msg.pop("word embeddings")
    # Deal with padding
    full_word_embed = padding_vocab(orig_word_embed)
    # Split into new tensor model parallel sizes
    out_word_embed = torch.chunk(
        full_word_embed, args.target_tensor_parallel_size, dim=0
    )
    # Make models for first pipeline stage and fill in embeddings
    post_process = args.target_pipeline_parallel_size == 1
    for tp_rank in range(args.target_tensor_parallel_size):
        model_ckpt = get_model_ckpt(model_ckpts, tp_rank, 0, vp_size, 0)
        set_weight_or_bias(
            model_ckpt, None, vp_size, 0, "word_embeddings", out_word_embed[tp_rank]
        )
    check_message(embeddings_msg)

    # Transformer layers
    # -------------------
    total_layer_num = 0
    for pp_rank in range(args.target_pipeline_parallel_size):
        # For later pipeline parallel ranks, make the new models
        if pp_rank > 0:
            post_process = pp_rank == args.target_pipeline_parallel_size - 1
        num_layers = get_num_layers_from_args(
            md.num_layers, pp_size, pp_rank, margs.hetero_pipeline_stage_split
        )
        for layer in range(num_layers):
            # weight
            msg = queue_get(f"transformer layer {total_layer_num}")

            # duplicated tensors
            input_norm_weight = msg.pop("input norm weight")
            if md.norm_has_bias:
                input_norm_bias = msg.pop("input norm bias")
            post_norm_weight = msg.pop("post norm weight")
            if md.norm_has_bias:
                post_norm_bias = msg.pop("post norm bias")
            if md.linear_bias:
                dense_bias = msg.pop("dense bias")
                mlp_l1_bias = msg.pop("mlp l1 bias")

            # Split up the parallel tensors
            qkv_weight = torch.chunk(
                msg.pop("qkv weight"), args.target_tensor_parallel_size, dim=0
            )
            dense_weight = torch.chunk(
                msg.pop("dense weight"), args.target_tensor_parallel_size, dim=1
            )
            mlp_l1_weight = torch.chunk(
                msg.pop("mlp l1 weight"), args.target_tensor_parallel_size, dim=1
            )

            # Special handling for swiglu
            if md.swiglu:
                mlp_l0_weight_W = torch.chunk(
                    msg.pop("mlp l0 weight W"), args.target_tensor_parallel_size, dim=0
                )
                mlp_l0_weight_V = torch.chunk(
                    msg.pop("mlp l0 weight V"), args.target_tensor_parallel_size, dim=0
                )
                mlp_l0_weight = [
                    torch.cat(weights, dim=0)
                    for weights in zip(mlp_l0_weight_W, mlp_l0_weight_V)
                ]
            else:
                mlp_l0_weight = torch.chunk(
                    msg.pop("mlp l0 weight"), args.target_tensor_parallel_size, dim=0
                )

            if md.linear_bias:
                qkv_main_bias = torch.chunk(
                    msg.pop("qkv bias"), args.target_tensor_parallel_size, dim=0
                )
                if md.swiglu:
                    mlp_l0_bias_W = torch.chunk(
                        msg.pop("mlp l0 bias W"),
                        args.target_tensor_parallel_size,
                        dim=0,
                    )
                    mlp_l0_bias_V = torch.chunk(
                        msg.pop("mlp l0 bias V"),
                        args.target_tensor_parallel_size,
                        dim=0,
                    )
                    mlp_l0_bias = [
                        torch.cat(bias, dim=0)
                        for bias in zip(mlp_l0_bias_W, mlp_l0_bias_V)
                    ]
                else:
                    mlp_l0_bias = torch.chunk(
                        msg.pop("mlp l0 bias"), args.target_tensor_parallel_size, dim=0
                    )

            # Save them to the model
            for tp_rank in range(args.target_tensor_parallel_size):
                model_ckpt = get_model_ckpt(model_ckpts, tp_rank, pp_rank, vp_size, 0)
                set_weight_or_bias(
                    model_ckpt,
                    layer,
                    vp_size,
                    0,
                    "input_norm",
                    input_norm_weight,
                )
                if md.norm_has_bias:
                    set_weight_or_bias(
                        model_ckpt,
                        layer,
                        vp_size,
                        0,
                        "input_norm",
                        input_norm_bias,
                        True,
                    )
                set_weight_or_bias(
                    model_ckpt,
                    layer,
                    vp_size,
                    0,
                    "query_key_value",
                    qkv_weight[tp_rank],
                )
                set_weight_or_bias(
                    model_ckpt, layer, vp_size, 0, "dense", dense_weight[tp_rank]
                )
                set_weight_or_bias(
                    model_ckpt,
                    layer,
                    vp_size,
                    0,
                    "post_attention_norm",
                    post_norm_weight,
                )
                if md.norm_has_bias:
                    set_weight_or_bias(
                        model_ckpt,
                        layer,
                        vp_size,
                        0,
                        "post_attention_norm",
                        post_norm_bias,
                        True,
                    )
                set_weight_or_bias(
                    model_ckpt,
                    layer,
                    vp_size,
                    0,
                    "dense_h_to_4h",
                    mlp_l0_weight[tp_rank],
                )
                set_weight_or_bias(
                    model_ckpt,
                    layer,
                    vp_size,
                    0,
                    "dense_4h_to_h",
                    mlp_l1_weight[tp_rank],
                )
                if md.linear_bias:
                    set_weight_or_bias(
                        model_ckpt,
                        layer,
                        vp_size,
                        0,
                        "query_key_value",
                        qkv_main_bias[tp_rank],
                        True,
                    )
                    set_weight_or_bias(
                        model_ckpt, layer, vp_size, 0, "dense", dense_bias, True
                    )
                    set_weight_or_bias(
                        model_ckpt,
                        layer,
                        vp_size,
                        0,
                        "dense_h_to_4h",
                        mlp_l0_bias[tp_rank],
                        True,
                    )
                    set_weight_or_bias(
                        model_ckpt,
                        layer,
                        vp_size,
                        0,
                        "dense_4h_to_h",
                        mlp_l1_bias,
                        True,
                    )
            check_message(msg)

            total_layer_num = total_layer_num + 1

        if post_process:
            msg = queue_get("final norm")
            final_norm_weight = msg.pop("weight")
            if md.norm_has_bias:
                final_norm_bias = msg.pop("bias")
            for tp_rank in range(args.target_tensor_parallel_size):
                model_ckpt = get_model_ckpt(model_ckpts, tp_rank, pp_rank, vp_size, 0)
                set_weight_or_bias(
                    model_ckpt,
                    None,
                    vp_size,
                    0,
                    "final_norm",
                    final_norm_weight,
                )
                if md.norm_has_bias:
                    set_weight_or_bias(
                        model_ckpt,
                        None,
                        vp_size,
                        0,
                        "final_norm",
                        final_norm_bias,
                        True,
                    )
                if pp_rank != 0 and not md.output_layer:
                    # Copy word embeddings to final pipeline rank
                    set_weight_or_bias(
                        model_ckpt,
                        None,
                        vp_size,
                        0,
                        "embedding",
                        out_word_embed[tp_rank],
                    )
            check_message(msg)

            if md.output_layer:
                msg = queue_get("output layer")
                orig_output_layer_weight = msg.pop("weight")
                full_output_layer_weight = padding_vocab(orig_output_layer_weight)
                output_layer_weight = torch.chunk(
                    full_output_layer_weight, args.target_tensor_parallel_size, dim=0
                )
                for tp_rank in range(args.target_tensor_parallel_size):
                    model_ckpt = get_model_ckpt(
                        model_ckpts, tp_rank, pp_rank, vp_size, 0
                    )
                    set_weight_or_bias(
                        model_ckpt,
                        None,
                        vp_size,
                        0,
                        "output_layer",
                        output_layer_weight[tp_rank],
                    )
                check_message(msg)

            msg = queue_get()
            if msg != "done":
                print("ERROR: got some more data but was expecting to be done")

        for tp_rank in range(args.target_tensor_parallel_size):
            set_optimizer_info(model_ckpts[tp_rank][pp_rank], optimizer, opt_param_scheduler, margs)
            save_model_ckpt(model_ckpts, model_ckpt_paths, tp_rank, pp_rank)
            print("Saved weight for tp_rank {} and pp_rank {}".format(tp_rank, pp_rank))

    print("Done!")
