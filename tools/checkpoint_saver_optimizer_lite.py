"""
This script is based on Megatron's checkpoint_saver_megatron.py, but it only saves the distributed optimizer state.
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
    get_optimizer_ckpt_paths,
    get_param_index_map_paths,
    get_num_layers_from_args,
    flatten_optimizer_ckpt,
    merge_optimizer_ckpt,
    remove_optimizer_tmp,
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
    assert tp_size == margs.tensor_model_parallel_size
    assert pp_size == margs.pipeline_model_parallel_size

    def get_optimizer_ckpt(
        optimizer_ckpts, state_key, tp_rank, pp_rank, vp_rank, dtype=torch.float32
    ):
        assert state_key in ["param", "exp_avg", "exp_avg_sq"]
        if optimizer_ckpts[tp_rank][pp_rank] is None:
            optimizer_ckpt = {}
            optimizer_ckpt[vp_rank] = {}
            optimizer_ckpt[vp_rank][dtype] = {
                state_key: {},
            }
            optimizer_ckpts[tp_rank][pp_rank] = optimizer_ckpt
            return optimizer_ckpts[tp_rank][pp_rank]
        else:
            return optimizer_ckpts[tp_rank][pp_rank]

    def set_optimizer_state(
        optimizer_ckpt,
        layer_num,
        vp_rank,
        state_key,
        param_key,
        val,
        bias=False,
        dtype=torch.float32,
    ):
        assert state_key in ["param", "exp_avg", "exp_avg_sq"]
        if param_key == "word_embeddings":
            full_key = "module.language_model.embedding.word_embeddings.weight"
        elif param_key == "input_layernorm":
            if not bias:
                full_key = f"module.language_model.encoder.layers.{layer_num}.input_layernorm.weight"
            else:
                full_key = f"module.language_model.encoder.layers.{layer_num}.input_layernorm.bias"
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
        elif param_key == "post_attention_layernorm":
            if not bias:
                full_key = f"module.language_model.encoder.layers.{layer_num}.post_attention_layernorm.weight"
            else:
                full_key = f"module.language_model.encoder.layers.{layer_num}.post_attention_layernorm.bias"
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
        elif param_key == "final_layernorm":
            if not bias:
                full_key = f"module.language_model.encoder.final_layernorm.weight"
            else:
                full_key = f"module.language_model.encoder.final_layernorm.bias"
        elif param_key == "output_layer":
            full_key = f"module.language_model.output_layer.weight"
        else:
            print(
                "[WARNING]: saver unrecognized state key: "
                + state_key
                + ", param_key "
                + param_key
            )
            exit(1)
        optimizer_ckpt[vp_rank][dtype][state_key][full_key] = val

    if not args.saver_mapping_from_start:
        param_index_map_paths = get_param_index_map_paths(md.load, tp_size, pp_size, md.iteration)
    else:
        param_index_map_paths = get_param_index_map_paths(margs.save, tp_size, pp_size, 0)
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
        margs.save, tp_size, pp_size, md.iteration
    )

    def save_optimizer_ckpt(
        optimizer_ckpts, optimizer_ckpt_paths, state_key, tp_rank, pp_rank
    ):
        assert state_key in ["param", "exp_avg", "exp_avg_sq"]
        if optimizer_ckpts[tp_rank][pp_rank] is not None:
            optimizer_ckpt_path = optimizer_ckpt_paths[tp_rank][pp_rank]
            optimizer_ckpt_dir = os.path.dirname(optimizer_ckpt_path)
            if not os.path.exists(optimizer_ckpt_dir):
                os.makedirs(optimizer_ckpt_dir)
            param_index_map = get_param_index_map(
                param_index_maps, param_index_map_paths, tp_rank, pp_rank
            )
            flatten_ckpt = flatten_optimizer_ckpt(
                optimizer_ckpts[tp_rank][pp_rank], param_index_map, vp_size, state_key
            )
            ckpt_name, ckpt_ext = os.path.splitext(optimizer_ckpt_path)
            save_path = ckpt_name + "_" + state_key + ckpt_ext
            print(f"Save optimizer state to {save_path}")
            torch.save(flatten_ckpt, save_path)
            # Delete the saved optimizer state
            optimizer_ckpt = optimizer_ckpts[tp_rank][pp_rank]
            optimizer_ckpts[tp_rank][pp_rank] = None
            del optimizer_ckpt
        else:
            raise Exception("optimizer_state ckpt is None")

    def padding_vocab(orig_word_embed):
        # Deal with padding
        if md.true_vocab_size is not None:
            # figure out what our padded vocab size is
            orig_vocab_size = orig_word_embed.shape[0]
            margs.padded_vocab_size = _vocab_size_with_padding(
                md.true_vocab_size, margs
            )

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

    def receive_message(state_key):
        optimizer_ckpts = [[None for _ in range(pp_size)] for _ in range(tp_size)]
        # Embeddings
        assert md.position_embedding_type != "learned_absolute"
        embeddings_msg = queue_get(f"embeddings {state_key}")
        orig_word_embed = embeddings_msg.pop(f"word embeddings {state_key}")
        # Deal with padding
        full_word_embed = padding_vocab(orig_word_embed)
        # Split into new tensor model parallel sizes
        out_word_embed = torch.chunk(
            full_word_embed, args.target_tensor_parallel_size, dim=0
        )
        # Make models for first pipeline stage and fill in embeddings
        post_process = args.target_pipeline_parallel_size == 1
        for tp_rank in range(args.target_tensor_parallel_size):
            optimizer_ckpt = get_optimizer_ckpt(optimizer_ckpts, state_key, tp_rank, 0, 0)
            set_optimizer_state(
                optimizer_ckpt,
                None,
                0,
                state_key,
                "word_embeddings",
                out_word_embed[tp_rank],
            )
        check_message(embeddings_msg)

        # Transformer layers
        # -------------------
        total_layer_num = 0
        for pp_rank in range(args.target_pipeline_parallel_size):
            # For later pipeline parallel ranks, make the new models
            if pp_rank > 0:
                post_process = pp_rank == args.target_pipeline_parallel_size - 1
            num_layers = get_num_layers_from_args(md.num_layers, pp_size, pp_rank)
            for layer in range(num_layers):
                # -----------------
                # main weight
                # -----------------
                msg = queue_get(f"transformer layer {total_layer_num} {state_key}")
                # duplicated tensors
                input_layernorm_weight = msg.pop(f"input layernorm weight {state_key}")
                if not md.apply_layernorm_rms:
                    input_layernorm_bias = msg.pop(f"input layernorm bias {state_key}")
                post_layernorm_weight = msg.pop(f"post layernorm weight {state_key}")
                if not md.apply_layernorm_rms:
                    post_layernorm_bias = msg.pop(f"post layernorm bias {state_key}")
                if md.linear_bias:
                    dense_bias = msg.pop(f"dense bias {state_key}")
                    mlp_l1_bias = msg.pop(f"mlp l1 bias {state_key}")

                # Split up the parallel tensors
                qkv_weight = torch.chunk(
                    msg.pop(f"qkv weight {state_key}"), args.target_tensor_parallel_size, dim=0
                )
                dense_weight = torch.chunk(
                    msg.pop(f"dense weight {state_key}"), args.target_tensor_parallel_size, dim=1
                )
                mlp_l1_weight = torch.chunk(
                    msg.pop(f"mlp l1 weight {state_key}"), args.target_tensor_parallel_size, dim=1
                )

                # Special handling for swiglu
                if md.swiglu:
                    mlp_l0_W_weight = torch.chunk(
                        msg.pop(f"mlp l0 weight W {state_key}"),
                        args.target_tensor_parallel_size,
                        dim=0,
                    )
                    mlp_l0_V_weight = torch.chunk(
                        msg.pop(f"mlp l0 weight V {state_key}"),
                        args.target_tensor_parallel_size,
                        dim=0,
                    )
                    mlp_l0_weight = [
                        torch.cat(weights, dim=0)
                        for weights in zip(mlp_l0_W_weight, mlp_l0_V_weight)
                    ]
                else:
                    mlp_l0_weight= torch.chunk(
                        msg.pop(f"mlp l0 weight {state_key}"),
                        args.target_tensor_parallel_size,
                        dim=0,
                    )

                if md.linear_bias:
                    qkv_bias = torch.chunk(
                        msg.pop(f"qkv bias {state_key}"), args.target_tensor_parallel_size, dim=0
                    )
                    if md.swiglu:
                        mlp_l0_bias_W = torch.chunk(
                            msg.pop(f"mlp l0 bias W {state_key}"),
                            args.target_tensor_parallel_size,
                            dim=0,
                        )
                        mlp_l0_bias_V = torch.chunk(
                            msg.pop(f"mlp l0 bias V {state_key}"),
                            args.target_tensor_parallel_size,
                            dim=0,
                        )
                        mlp_l0_bias = [
                            torch.cat(main_bias, dim=0)
                            for main_bias in zip(mlp_l0_bias_W, mlp_l0_bias_V)
                        ]
                    else:
                        mlp_l0_bias = torch.chunk(
                            msg.pop(f"mlp l0 bias {state_key}"),
                            args.target_tensor_parallel_size,
                            dim=0,
                        )

                # Save them to the model
                for tp_rank in range(args.target_tensor_parallel_size):
                    optimizer_ckpt = get_optimizer_ckpt(
                        optimizer_ckpts, state_key, tp_rank, pp_rank, 0
                    )
                    set_optimizer_state(
                        optimizer_ckpt,
                        layer,
                        0,
                        state_key,
                        "input_layernorm",
                        input_layernorm_weight,
                    )
                    if not md.apply_layernorm_rms:
                        set_optimizer_state(
                            optimizer_ckpt,
                            layer,
                            0,
                            state_key,
                            "input_layernorm",
                            input_layernorm_bias,
                            True,
                        )
                    set_optimizer_state(
                        optimizer_ckpt,
                        layer,
                        0,
                        state_key,
                        "query_key_value",
                        qkv_weight[tp_rank],
                    )
                    set_optimizer_state(
                        optimizer_ckpt,
                        layer,
                        0,
                        state_key,
                        "dense",
                        dense_weight[tp_rank],
                    )
                    set_optimizer_state(
                        optimizer_ckpt,
                        layer,
                        0,
                        state_key,
                        "post_attention_layernorm",
                        post_layernorm_weight,
                    )
                    if not md.apply_layernorm_rms:
                        set_optimizer_state(
                            optimizer_ckpt,
                            layer,
                            0,
                            state_key,
                            "post_attention_layernorm",
                            post_layernorm_bias,
                            True,
                        )
                    set_optimizer_state(
                        optimizer_ckpt,
                        layer,
                        0,
                        state_key,
                        "dense_h_to_4h",
                        mlp_l0_weight[tp_rank],
                    )
                    set_optimizer_state(
                        optimizer_ckpt,
                        layer,
                        0,
                        state_key,
                        "dense_4h_to_h",
                        mlp_l1_weight[tp_rank],
                    )
                    if md.linear_bias:
                        set_optimizer_state(
                            optimizer_ckpt,
                            layer,
                            0,
                            state_key,
                            "query_key_value",
                            qkv_bias[tp_rank],
                            True,
                        )
                        set_optimizer_state(
                            optimizer_ckpt,
                            layer,
                            0,
                            state_key,
                            "dense",
                            dense_bias,
                            True,
                        )
                        set_optimizer_state(
                            optimizer_ckpt,
                            layer,
                            0,
                            state_key,
                            "dense_h_to_4h",
                            mlp_l0_bias[tp_rank],
                            True,
                        )
                        set_optimizer_state(
                            optimizer_ckpt,
                            layer,
                            0,
                            state_key,
                            "dense_4h_to_h",
                            mlp_l1_bias,
                            True,
                        )
                check_message(msg)

                total_layer_num = total_layer_num + 1

            if post_process:
                msg = queue_get(f"final layernorm {state_key}")
                final_layernorm_weight = msg.pop(f"weight {state_key}")
                if not md.apply_layernorm_rms:
                    final_layernorm_main_bias = msg.pop(f"bias {state_key}")
                for tp_rank in range(args.target_tensor_parallel_size):
                    # inspect_dict(optimizer_ckpts[tp_rank][pp_rank])
                    optimizer_ckpt = get_optimizer_ckpt(
                        optimizer_ckpts, state_key, tp_rank, pp_rank, 0
                    )
                    set_optimizer_state(
                        optimizer_ckpt,
                        None,
                        0,
                        state_key,
                        "final_layernorm",
                        final_layernorm_weight,
                    )
                    if not md.apply_layernorm_rms:
                        set_optimizer_state(
                            optimizer_ckpt,
                            None,
                            0,
                            state_key,
                            "final_layernorm",
                            final_layernorm_main_bias,
                            True,
                        )
                    if pp_rank != 0 and not md.output_layer:
                        # Copy word embeddings to final pipeline rank
                        set_optimizer_state(
                            optimizer_ckpt,
                            None,
                            0,
                            state_key,
                            "embedding",
                            out_word_embed[tp_rank],
                        )
                check_message(msg)


                if md.output_layer:
                    msg = queue_get(f"output layer {state_key}")
                    # Deal with padding
                    orig_output_layer_weight = msg.pop(f"weight {state_key}")
                    full_output_layer_weight = padding_vocab(
                        orig_output_layer_weight
                    )
                    output_layer_weight = torch.chunk(
                        full_output_layer_weight,
                        args.target_tensor_parallel_size,
                        dim=0,
                    )
                    for tp_rank in range(args.target_tensor_parallel_size):
                        optimizer_ckpt = get_optimizer_ckpt(
                            optimizer_ckpts, state_key, tp_rank, pp_rank, 0
                        )
                        set_optimizer_state(
                            optimizer_ckpt,
                            None,
                            0,
                            state_key,
                            "output_layer",
                            output_layer_weight[tp_rank],
                        )
                    check_message(msg)


            for tp_rank in range(args.target_tensor_parallel_size):
                save_optimizer_ckpt(
                    optimizer_ckpts, optimizer_ckpt_paths, state_key, tp_rank, pp_rank
                )
                print(
                    "Saved model and optimizer state for tp_rank {} and pp_rank {}".format(
                        tp_rank, pp_rank
                    )
                )
    
    receive_message("param")

    receive_message("exp_avg")

    receive_message("exp_avg_sq")

    msg = queue_get()
    if msg != "done":
        print("ERROR: got some more data but was expecting to be done")
    else:
        print("Conversion optimizer state is done!")

    # ------- merge optimizer ckpts -------
    for tp_rank in range(tp_size):
        for pp_rank in range(pp_size):
            merge_optimizer_ckpt(optimizer_ckpt_paths[tp_rank][pp_rank], vp_size)

    # ------- remove the tmp optimizer ckpts -------
    if args.del_tmp:
        for tp_rank in range(tp_size):
            for pp_rank in range(pp_size):
                remove_optimizer_tmp(optimizer_ckpt_paths[tp_rank][pp_rank])
