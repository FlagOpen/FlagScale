import os
import sys
import importlib

import torch


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron saver')

    # group.add_argument('--megatron-path', type=str, default=None,
    #                 help='Base directory of deepspeed repository')
    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-pipeline-parallel-size', type=int,
                       help='Target tensor model parallel size, default to the pipeline parall size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument("--target-expert-parallel-size", type=int,
                      help='The tensor model parallel size of the converted checkpoint. '
                           'Only used when converting a Transformers checkpoint to a Megatron checkpoint.')
    group.add_argument("--target-params-dtype", type=str, default=None,
                       help='The dtype of the converted checkpoint. '
                            'Only used when converting a Transformers checkpoint to a Megatron checkpoint.')


def save_checkpoint(queue, args):
    # Search in directory above this
    root_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir))
    sys.path.append(os.path.join(root_path, "megatron"))

    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.arguments import parse_args, validate_args
        from megatron.checkpointing import save_checkpoint, get_checkpoint_name
        from megatron.global_vars import set_global_variables, get_args
        from megatron.tokenizer.tokenizer import _vocab_size_with_padding
        from megatron import fused_kernels
        from megatron.core import mpu
        from megatron.core.tensor_parallel.random import (
                _CUDA_RNG_STATE_TRACKER, _DATA_PARALLEL_RNG_TRACKER_NAME, 
                _EXPERT_PARALLEL_RNG_TRACKER_NAME, _MODEL_PARALLEL_RNG_TRACKER_NAME
            )
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(msg):
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            exit(1)

    md = queue.get()

    if args.target_tensor_parallel_size is None:
        if hasattr(md, 'previous_tensor_parallel_size'):
            args.target_tensor_parallel_size = md.previous_tensor_parallel_size
        else:
            print("loader did not provide a tensor parallel size and --target-tensor-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_tensor_parallel_size = 1

    if args.target_pipeline_parallel_size is None:
        if hasattr(md, 'previous_pipeline_parallel_size'):
            args.target_pipeline_parallel_size = md.previous_pipeline_parallel_size
        else:
            print("loader did not provide a pipeline parallel size and --target-pipeline-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_pipeline_parallel_size = 1

    if args.target_expert_parallel_size is None:
        if hasattr(md, 'previous_expert_parallel_size'):
            args.target_expert_parallel_size = md.previous_expert_parallel_size
        else:
            print("loader did not provide a pipeline parallel size and --target-pipeline-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_expert_parallel_size = 1

    os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size * args.target_pipeline_parallel_size * args.target_expert_parallel_size}'

    # We want all arguments to come from us
    sys.argv = [
        'script.py',
        '--num-layers', str(md.num_layers),
        '--hidden-size', str(md.hidden_size),
        '--seq-length', str(md.seq_length),
        '--num-attention-heads', str(md.num_attention_heads),
        '--max-position-embeddings', str(md.max_position_embeddings),
        '--position-embedding-type', str(md.position_embedding_type),
        '--tensor-model-parallel-size', str(args.target_tensor_parallel_size),
        '--pipeline-model-parallel-size', str(args.target_pipeline_parallel_size),
        '--expert-model-parallel-size', str(args.target_expert_parallel_size),
        '--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion',
        '--no-bias-dropout-fusion',
        '--no-async-tensor-model-parallel-allreduce',
        '--use-cpu-initialization',
        "--use-mcore-models",
        '--micro-batch-size', '1',
        '--no-load-optim',
        '--no-load-rng',
        '--no-save-optim',
        '--no-save-rng',
        '--no-initialization',
        '--save-interval', '1',
        '--save', args.save_dir
    ]

    if md.make_vocab_size_divisible_by is not None:
        sys.argv.extend(['--make-vocab-size-divisible-by', str(md.make_vocab_size_divisible_by)])
    if md.output_layer:
        sys.argv.append('--untie-embeddings-and-output-weights')
    if not md.add_bias_linear:
        sys.argv.append('--disable-bias-linear')
    if md.add_qkv_bias:
        sys.argv.append('--add-qkv-bias')
    if md.model_type == 'BERT' and not md.bert_binary_head:
        sys.argv.append('--bert-no-binary-head')
    if args.target_expert_parallel_size > 1:
        sys.argv.append('--sequence-parallel')
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    if md.params_dtype == torch.float16:
        sys.argv.append('--fp16')
    elif md.params_dtype == torch.bfloat16:
        sys.argv.append('--bf16')

    margs = parse_args()
    if hasattr (md, 'checkpoint_args'):
        # These are arguments that we are either changing, or cause problems for validation if they are set
        # Note that some of these deal with T5 so will need to be changed if we support T5.
        args_to_keep = ['tensor_model_parallel_size', 'pipeline_model_parallel_size', 'world_size', 'params_dtype',
                        'num_layers_per_virtual_pipeline_stage', 'virtual_pipeline_model_parallel_size',
                        'masked_softmax_fusion', 'bias_gelu_fusion', 'bias_dropout_fusion',
                        'sequence_parallel', 'async_tensor_model_parallel_allreduce',
                        'no_load_optim', 'no_load_rng', 'no_save_optim', 'no_save_rng',
                        'vocab_file', 'tokenizer_model', 'expert_model_parallel_size',
                        'save_interval', 'save', 'use_mcore_models',
                        'perform_initialization', 'use_cpu_initialization',
                        'recompute_granularity', 'recompute_num_layers', 'recompute_method',
                        'encoder_num_layers', 'encoder_seq_length',
                        'distribute_saved_activations',
                        'train_iters', 'lr_decay_iters', 'lr_warmup_iters', 'lr_warmup_fraction',
                        'start_weight_decay', 'end_weight_decay']
        for arg, value in vars(md.checkpoint_args).items():
            if arg in args_to_keep:
                continue
            if not hasattr(margs, arg):
                print(f"Checkpoint had argument {arg} but new arguments does not have this.")
                continue
            if getattr(margs, arg) != value:
                print(f"Overwriting default {arg} value {getattr(margs, arg)} with value from checkpoint {value}.")
                setattr(margs, arg, value)
    validate_args(margs)

    # megatron args
    set_global_variables(margs, build_tokenizer=False)
    margs = get_args()

    if hasattr(md, 'consumed_train_samples'):
        margs.consumed_train_samples = md.consumed_train_samples
        margs.consumed_valid_samples = md.consumed_valid_samples
        print(f"Setting consumed_train_samples to {margs.consumed_train_samples}"
              f" and consumed_valid_samples to {margs.consumed_valid_samples}")
    else:
        print("consumed_train_samples not provided.")

    try:
        model_plugin = importlib.import_module(args.model_type + ".model")
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please check model_type or model.py")

    def get_models(count, dtype):
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        models = [model_plugin.get_mg_model(dtype, pre_process, post_process) for _ in range(count)]
        return models

    margs.model_type = model_plugin.model_type

    # fake initializing distributed
    tp_size = args.target_tensor_parallel_size
    pp_size = args.target_pipeline_parallel_size
    ep_size = args.target_expert_parallel_size

    mpu.set_tensor_model_parallel_world_size(tp_size)
    mpu.set_pipeline_model_parallel_world_size(pp_size)
    mpu.set_expert_model_parallel_world_size(ep_size)
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)

    # fused kernel
    fused_kernels.load(margs)

    # random
    _CUDA_RNG_STATE_TRACKER.reset()
    torch.cuda.manual_seed(42)
    _CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, 43)
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 44)
    _CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, 45)

    # embedding
    embeddings_msg = queue_get("embeddings")
    pos_embed = None
    if md.position_embedding_type == 'learned_absolute':
        pos_embed = embeddings_msg.pop("position embeddings")
    orig_word_embed = embeddings_msg.pop("word embeddings")
    check_message(embeddings_msg)

    # padding vocab_size
    if md.true_vocab_size is not None:
        # figure out what our padded vocab size is
        orig_vocab_size = orig_word_embed.shape[0]
        margs.padded_vocab_size = _vocab_size_with_padding(md.true_vocab_size, margs)

        # Cut out extra padding we don't need
        if orig_vocab_size > margs.padded_vocab_size:
            full_word_embed = orig_word_embed[0:margs.padded_vocab_size,:]

        # Expanding embedding to larger size by replicating final entry
        elif orig_vocab_size < margs.padded_vocab_size:
            padding_size = margs.padded_vocab_size - orig_vocab_size

            full_word_embed = torch.cat((
                orig_word_embed,
                orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1)))

        # Same size!
        else:
            full_word_embed = orig_word_embed
    else:
        print("Original vocab size not specified, leaving embedding table as-is. "
              "If you've changed the tensor parallel size this could cause problems.")
        margs.padded_vocab_size = orig_word_embed.shape[0]
        full_word_embed = orig_word_embed

    # process world embedding in first pp stage
    mpu.set_pipeline_model_parallel_rank(0)
    models = get_models(tp_size * ep_size, md.params_dtype)
    out_word_embed = torch.chunk(full_word_embed, tp_size, dim=0)
    for tp_ep_rank, model in enumerate(models):
        tp_rank = tp_ep_rank % tp_size
        model.embedding.word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
        if pos_embed is not None:
            model.embedding.position_embeddings.weight.data.copy_(pos_embed)
        else:
            assert not hasattr(model.embedding, "position_embeddings")

    # process transformer layer
    total_layer_num = 0
    for pp_rank in range(pp_size):

        mpu.set_pipeline_model_parallel_rank(pp_rank)

        if pp_rank > 0:
            models = get_models(tp_size * ep_size, md.params_dtype)

        for layer_id in range(len(models[0].decoder.layers)):
            msg = queue_get(f"transformer layer {total_layer_num}")

            # process self attention
            post_norm_weight = msg.pop("post norm weight")
            qkv_weight = torch.chunk(msg.pop("qkv weight"), tp_size, dim=0)
            proj_weight = torch.chunk(msg.pop("proj weight"), tp_size, dim=1)
            if md.add_qkv_bias or md.add_bias_linear:
                qkv_bias = torch.chunk(msg.pop("qkv bias"), tp_size, dim=0)
                proj_bias = msg.pop("proj bias")
            # split
            for tp_ep_rank, model in enumerate(models):
                tp_rank = tp_ep_rank % tp_size
                layer = model.decoder.layers[layer_id]
                layer.self_attention.linear_qkv.weight.data.copy_(qkv_weight[tp_rank])
                layer.self_attention.linear_proj.weight.data.copy_(proj_weight[tp_rank])
                layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(post_norm_weight)
                if md.add_qkv_bias or md.add_bias_linear:
                    layer.self_attention.linear_qkv.bias.data.copy_(qkv_bias[tp_rank])
                    layer.self_attention.linear_proj.bias.data.copy_(proj_bias)

            # process mlp
            pre_mlp_layernorm = msg.pop("pre norm weight")
            if md.num_experts:
                router_weight = msg.pop("router weight")
                num_local_experts = md.num_experts // ep_size
                for expert_id in range(num_local_experts):
                    for ep_rank in range(ep_size):
                        global_expert_id = ep_rank * num_local_experts + expert_id
                        # weight
                        expert_l1_weight = torch.chunk(msg.pop(f"expert{global_expert_id} l1 weight"), tp_size, dim=1)
                        if md.swiglu:
                            expert_l0_weight_W = torch.chunk(msg.pop(f"expert{global_expert_id} l0 weight W"), tp_size, dim=0)
                            expert_l0_weight_V = torch.chunk(msg.pop(f"expert{global_expert_id} l0 weight V"), tp_size, dim=0)
                            expert_l0_weight = [torch.cat(weights, dim=0) for weights in zip(expert_l0_weight_W, expert_l0_weight_V)]
                        else:
                            expert_l0_weight = torch.chunk(msg.pop(f"expert{global_expert_id} l0 weight"), tp_size, dim=0)
                        # bias
                        if md.add_bias_linear:
                            expert_l1_bias = msg.pop(f"expert{global_expert_id} l1 bias")
                            if md.swiglu:
                                expert_l0_bias_W = torch.chunk(msg.pop(f"expert{global_expert_id} l0 bias W"), tp_size, dim=0)
                                expert_l0_bias_V = torch.chunk(msg.pop(f"expert{global_expert_id} l0 bias V"), tp_size, dim=0)
                                expert_l0_bias = [torch.cat(bias, dim=0) for bias in zip(expert_l0_bias_W, expert_l0_bias_V)]
                            else:
                                expert_l0_bias = torch.chunk(msg.pop(f"expert{global_expert_id} l0 bias"), tp_size, dim=0)
                        # copy
                        for tp_rank in range(tp_size):
                            tp_ep_rank = ep_rank * tp_size + tp_rank
                            layer = models[tp_ep_rank].decoder.layers[layer_id]
                            layer.mlp.router.weight.data.copy_(router_weight)
                            layer.pre_mlp_layernorm.weight.data.copy_(pre_mlp_layernorm)
                            expert = layer.mlp.experts.local_experts[expert_id]
                            expert.linear_fc1.weight.data.copy_(expert_l0_weight[tp_rank])
                            expert.linear_fc2.weight.data.copy_(expert_l1_weight[tp_rank])
                            if md.add_bias_linear:
                                expert.linear_fc1.bias.data.copy_(expert_l0_bias[tp_rank])
                                expert.linear_fc2.bias.data.copy_(expert_l1_bias)
            else:
                # weight
                mlp_l1_weight = torch.chunk(msg.pop("mlp l1 weight"), tp_size, dim=1)
                if md.swiglu:
                    mlp_l0_weight_W = torch.chunk(msg.pop("mlp l0 weight W"), tp_size, dim=0)
                    mlp_l0_weight_V = torch.chunk(msg.pop("mlp l0 weight V"), tp_size, dim=0)
                    mlp_l0_weight = [torch.cat(weights, dim=0) for weights in zip(mlp_l0_weight_W, mlp_l0_weight_V)]
                else:
                    mlp_l0_weight = torch.chunk(msg.pop("mlp l0 weight"), tp_size, dim=0)
                # bias
                if md.add_bias_linear:
                    mlp_l1_bias = msg.pop("mlp l1 bias")
                    if md.swiglu:
                        mlp_l0_bias_W = torch.chunk(msg.pop("mlp l0 bias W"), tp_size, dim=0)
                        mlp_l0_bias_V = torch.chunk(msg.pop("mlp l0 bias V"), tp_size, dim=0)
                        mlp_l0_bias = [torch.cat(bias, dim=0) for bias in zip(mlp_l0_bias_W, mlp_l0_bias_V)]
                    else:
                        mlp_l0_bias = torch.chunk(msg.pop("mlp l0 bias"), tp_size, dim=0)
                # split and copy
                for tp_ep_rank, model in enumerate(models):
                    tp_rank = tp_ep_rank % tp_size
                    layer = model.decoder.layers[layer_id]
                    layer.pre_mlp_layernorm.weight.data.copy_(pre_mlp_layernorm)
                    layer.mlp.linear_fc2.weight.data.copy_(mlp_l1_weight[tp_rank])
                    layer.mlp.linear_fc1.weight.data.copy_(mlp_l0_weight[tp_rank])
                    if md.add_bias_linear:
                        layer.mlp.linear_fc1.bias.data.copy_(mlp_l0_bias[tp_rank])
                        layer.mlp.linear_fc2.bias.data.copy_(mlp_l1_bias)

            total_layer_num = total_layer_num + 1
            check_message(msg)

        # process final layernorm and linear
        if pp_rank == pp_size - 1:
            msg = queue_get("final norm")
            final_norm_weight = msg.pop("weight")
            if md.norm_has_bias:
                final_norm_bias = msg.pop("bias")
            for tp_ep_rank, model in enumerate(models):
                model.decoder.final_layernorm.weight.data.copy_(final_norm_weight)
                if md.norm_has_bias:
                    model.decoder.final_layernorm.bias.data.copy_(final_norm_bias)
            check_message(msg)

            if md.output_layer:
                msg = queue_get("output layer")
                assert hasattr(models[0], 'output_layer'), "ERROR: got an output layer, but model does not have one"
                orig_output_layer_weight = msg.pop("weight")
                if md.true_vocab_size is not None:
                    # figure out what our padded vocab size is
                    orig_output_layer_size = orig_output_layer_weight.shape[0]
                    margs.padded_vocab_size = _vocab_size_with_padding(md.true_vocab_size, margs)

                    # Cut out extra padding we don't need
                    if orig_output_layer_size > margs.padded_vocab_size:
                        full_output_layer_weight = orig_output_layer_weight[0:margs.padded_vocab_size,:]

                    # Expanding embedding to larger size by replicating final entry
                    elif orig_output_layer_size < margs.padded_vocab_size:
                        padding_size = margs.padded_vocab_size - orig_output_layer_size

                        full_output_layer_weight = torch.cat((
                            orig_output_layer_weight,
                            orig_output_layer_weight[-1].unsqueeze(0).expand(padding_size, -1)))

                    # Same size!
                    else:
                        full_output_layer_weight = orig_output_layer_weight
                else:
                    print("Original vocab size not specified, leaving embedding table as-is. "
                          "If you've changed the tensor parallel size this could cause problems.")
                    margs.padded_vocab_size = orig_output_layer_weight.shape[0]
                    full_output_layer_weight = orig_output_layer_weight

                output_layer_weight = torch.chunk(full_output_layer_weight, tp_size, dim=0)
                for tp_ep_rank, model in enumerate(models):
                    tp_rank = tp_ep_rank % tp_size
                    model.output_layer.weight.data.copy_(output_layer_weight[tp_rank])

            msg = queue_get()
            if msg != "done":
                print("ERROR: got some more data but was expecting to be done")

        for tp_ep_rank, model in enumerate(models):
            tp_rank = tp_ep_rank % tp_size
            ep_rank = tp_ep_rank // tp_size
            mpu.set_tensor_model_parallel_rank(tp_rank)
            mpu.set_expert_model_parallel_rank(ep_rank)
            checkpoint_name = get_checkpoint_name(margs.save, md.iteration)
            print("save to:", checkpoint_name)
            save_checkpoint(md.iteration, [model], None, None,
                            num_floating_point_operations_so_far=0)

    print("Done!!!")
