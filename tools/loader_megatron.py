import os
import sys
import json
import types
import importlib

import torch


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')

def _load_checkpoint(queue, args):

    # Search in directory above this
    root_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir))
    sys.path.append(os.path.join(root_path, "megatron"))

    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.arguments import parse_args, validate_args
        from megatron.global_vars import set_global_variables
        from megatron.checkpointing import load_args_from_checkpoint, load_checkpoint
        from megatron.model import module
        from megatron.core import mpu
        from megatron import fused_kernels
        from megatron.core.tensor_parallel.random import (
                _CUDA_RNG_STATE_TRACKER, _DATA_PARALLEL_RNG_TRACKER_NAME, 
                _EXPERT_PARALLEL_RNG_TRACKER_NAME, _MODEL_PARALLEL_RNG_TRACKER_NAME
            )
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us
    sys.argv = [
        'script.py',
        '--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion',
        '--no-bias-dropout-fusion',
        '--no-async-tensor-model-parallel-allreduce',
        '--use-cpu-initialization',
        '--micro-batch-size', '1',
        '--no-load-optim',
        '--no-load-rng',
        '--use-mcore-models',
        '--no-save-optim',
        '--no-save-rng',
        '--no-initialization',
        '--load', args.load_dir
    ]

    margs = parse_args()
    margs, checkpoint_args = load_args_from_checkpoint(margs)

    def _set_arg(arg_name):
        ckpt_value = getattr(checkpoint_args, arg_name, None)
        setattr(margs, arg_name, ckpt_value)

    _set_arg("expert_model_parallel_size")
    _set_arg("num_experts")
    _set_arg("sequence_parallel")

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size * margs.expert_model_parallel_size
    if margs.expert_model_parallel_size > 1:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
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

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('expert_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('iteration')
    check_for_arg('bert_binary_head')
    check_for_arg('params_dtype')
    check_for_arg('swiglu', False)
    check_for_arg('disable_bias_linear', False)
    check_for_arg('add_qkv_bias', getattr(margs, "add_bias_linear_qkv", True))

    try:
        model_plugin = importlib.import_module(args.model_type + ".model")
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please check model_type or model.py")

    margs.consumed_train_samples = 0
    margs.consumed_valid_samples = 0
    margs.model_type = model_plugin.model_type
    module.MegatronModule.embedding_warning_printed = True

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    def get_models(count, dtype):
        # for one pp stage
        tp_size = margs.tensor_model_parallel_size
        pp_size = margs.pipeline_model_parallel_size
        ep_size = margs.expert_model_parallel_size
        vp_size = margs.virtual_pipeline_model_parallel_size or 1

        models = [[] for _ in range(vp_size)]
        for rank_id in range(count):
            tp_rank = rank_id % tp_size
            ep_rank = rank_id // tp_size
            mpu.set_tensor_model_parallel_rank(tp_rank)
            mpu.set_expert_model_parallel_rank(ep_rank)
            if pp_size > 1 and vp_size > 1:
                model_ = []
                for vp_rank in range(vp_size):
                    mpu.set_virtual_pipeline_model_parallel_rank(vp_rank)
                    # Set pre_process and post_process only after virtual rank is set.
                    pre_process = mpu.is_pipeline_first_stage()
                    post_process = mpu.is_pipeline_last_stage()
                    this_model = model_plugin.get_mg_model(dtype, pre_process, post_process)
                    model_.append(this_model)
            else:
                pre_process = mpu.is_pipeline_first_stage()
                post_process = mpu.is_pipeline_last_stage() 
                model_ = [model_plugin.get_mg_model(dtype, pre_process, post_process)]

            load_checkpoint(model_, None, None)

            for vp_rank in range(vp_size):
                models[vp_rank].append(model_[vp_rank])

        return models

    # fake initializing distributed
    set_global_variables(margs, build_tokenizer=False)
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    ep_size = margs.expert_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size or 1

    mpu.set_tensor_model_parallel_world_size(tp_size)
    mpu.set_pipeline_model_parallel_world_size(pp_size)
    mpu.set_expert_model_parallel_world_size(ep_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(vp_size)
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)
    mpu.set_virtual_pipeline_model_parallel_rank(0)

    # fused kernel
    fused_kernels.load(margs)

    # random
    _CUDA_RNG_STATE_TRACKER.reset()
    torch.cuda.manual_seed(42)
    _CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, 43)
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 44)
    _CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, 45)

    # Get true (non-padded) vocab size
    if args.true_vocab_size is not None:
        true_vocab_size = args.true_vocab_size
    elif args.vocab_file is not None:
        vocab = json.load(open(args.vocab_file))
        true_vocab_size = len(vocab)
        if args.true_vocab_size is not None and true_vocab_size != args.true_vocab_size:
            print("Both --true-vocab-size and --vocab-file specified and the vocab size does not match, aborting.")
            queue.put("exit")
            exit(1)
    else:
        true_vocab_size = None

    # Layernorm has bias; RMSNorm does not.
    if hasattr(checkpoint_args, 'normalization'):
        norm_has_bias = checkpoint_args.normalization == "LayerNorm"
    else:
        # older models only supported LayerNorm
        norm_has_bias = True

    # metadata
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.num_experts = margs.num_experts
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    # md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.add_bias_linear = margs.add_bias_linear
    md.add_qkv_bias = margs.add_qkv_bias
    md.norm_has_bias = norm_has_bias
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.previous_expert_parallel_size = margs.expert_model_parallel_size
    md.true_vocab_size = true_vocab_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = checkpoint_args
    md.consumed_train_samples = margs.consumed_train_samples
    md.consumed_valid_samples = margs.consumed_valid_samples
    queue.put(md)

    # Get first pipe stage
    mpu.set_pipeline_model_parallel_rank(0)
    all_models = [get_models(tp_size * ep_size, md.params_dtype)]
    models = all_models[0][0] # pp0vpp0

    # Send embeddings
    word_embeddings = []
    complete_tp_ranks = []
    for tp_ep_rank, model in enumerate(models):
        tp_rank = tp_ep_rank % tp_size
        if tp_rank in complete_tp_ranks:
            continue
        complete_tp_ranks.append(tp_rank)
        word_embeddings.append(model.embedding.word_embeddings.weight.data)
    message = {"word embeddings": torch.cat(word_embeddings, dim=0)}
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = models[0].embedding.position_embeddings.weight.data
    else:
        assert not hasattr(models[0].embedding, 'position_embeddings')
    queue_put("embeddings", message)

    # Send transformer layer
    total_layer_num = 0
    for vp_rank in range(vp_size):
        mpu.set_virtual_pipeline_model_parallel_rank(vp_rank)
        for pp_rank in range(pp_size):
            mpu.set_pipeline_model_parallel_rank(pp_rank)

            if pp_rank > 0 and vp_rank == 0:
                all_models.append(get_models(tp_size * ep_size, md.params_dtype))

            models = all_models[pp_rank][vp_rank]
            for layer_id in range(len(models[0].decoder.layers)):
                message = {}

                # get self-attention tensors
                qkv_weight = []
                qkv_bias = []
                proj_weight = []
                proj_bias = None
                post_norm_weight = None
                post_norm_bias = None
                complete_tp_ranks = []
                for tp_ep_rank, model in enumerate(models):
                    tp_rank = tp_ep_rank % tp_size
                    if tp_rank in complete_tp_ranks:
                        continue
                    complete_tp_ranks.append(tp_rank)
                    layer = model.decoder.layers[layer_id]
                    qkv_weight.append(layer.self_attention.linear_qkv.weight.data)
                    proj_weight.append(layer.self_attention.linear_proj.weight.data)
                    post_norm_weight = layer.self_attention.linear_qkv.layer_norm_weight.data
                    if md.norm_has_bias:
                        post_norm_bias = layer.self_attention.linear_qkv.layer_norm_bias.data
                    if md.add_qkv_bias or md.add_bias_linear:
                        qkv_bias.append(layer.self_attention.linear_qkv.bias.data)
                        proj_bias = layer.self_attention.linear_proj.bias.data

                # concat self-attention tensors
                message["post norm weight"] = post_norm_weight
                if md.norm_has_bias:
                    message["post norm bias"] = post_norm_bias
                message["qkv weight"] = torch.cat(qkv_weight, dim=0) 
                message["proj weight"] = torch.cat(proj_weight, dim=1)
                if md.add_qkv_bias or md.add_bias_linear:
                    message["qkv bias"] = torch.cat(qkv_bias, dim=0)
                    message["proj bias"] = proj_bias

                # get pre-norm tensors
                layer = models[0].decoder.layers[layer_id]
                message["pre norm weight"] = layer.pre_mlp_layernorm.weight.data
                if md.norm_has_bias:
                    message["pre norm bias"] = layer.pre_mlp_layernorm.bias.data

                if md.num_experts:
                    # get experts tensors
                    message["router weight"] = layer.mlp.router.weight.data
                    num_local_experts = md.num_experts // ep_size
                    for expert_id in range(num_local_experts):
                        for ep_rank in range(ep_size):
                            global_expert_id = num_local_experts * ep_rank + expert_id
                            l0_weight = []
                            l0_bias = []
                            l1_weight = []
                            l1_bias = None
                            for tp_rank in range(tp_size):
                                tp_ep_rank = ep_rank * ep_size + tp_rank
                                layer = models[tp_ep_rank].decoder.layers[layer_id]
                                expert = layer.mlp.experts.local_experts[expert_id]

                                l0_weight.append(expert.linear_fc1.weight.data)
                                l1_weight.append(expert.linear_fc2.weight.data)
                                if md.add_bias_linear:
                                    l0_bias.append(expert.linear_fc1.bias.data)
                                    l1_bias = expert.linear_fc2.bias.data

                            # concat expert tensors
                            message[f"expert{global_expert_id} l1 weight"] = torch.cat(l1_weight, dim=1)
                            if md.swiglu:
                                for tp_rank in range(tp_size):
                                    l0_weight[tp_rank] = torch.chunk(l0_weight[tp_rank], 2, dim=0)
                                message[f"expert{global_expert_id} l0 weight W"] = torch.cat([w[0] for w in l0_weight], dim=0)
                                message[f"expert{global_expert_id} l0 weight V"] = torch.cat([w[1] for w in l0_weight], dim=0)
                            else:
                                message[f"expert{global_expert_id} l0 weight"] = torch.cat(l0_weight, dim=0)

                            if md.add_bias_linear:
                                message[f"expert{global_expert_id} l1 bias"] = l1_bias
                                if md.swiglu:
                                    for tp_rank in range(tp_size):
                                        l0_bias[tp_rank] = torch.chunk(l0_bias[tp_rank], 2, dim=0)
                                    message[f"expert{global_expert_id} l0 bias W"] = torch.cat([b[0] for b in l0_bias],dim=0)
                                    message[f"expert{global_expert_id} l0 bias V"] = torch.cat([b[1] for b in l0_bias],dim=0)
                                else:
                                    message[f"expert{global_expert_id} l0 bias"] = torch.cat(l0_bias, dim=0)
                else:
                    # get mlp tensors
                    l0_weight = []
                    l0_bias = []
                    l1_weight = []
                    l1_bias = None
                    complete_tp_ranks = []
                    for tp_ep_rank, model in enumerate(models):
                        tp_rank = tp_ep_rank % tp_size
                        if tp_rank in complete_tp_ranks:
                            continue
                        complete_tp_ranks.append(tp_rank)
                        layer = model.decoder.layers[layer_id]
                        l0_weight.append(layer.mlp.linear_fc1.weight.data)
                        l1_weight.append(layer.mlp.linear_fc2.weight.data)
                        if md.add_bias_linear:
                            l0_bias.append(layer.mlp.linear_fc1.bias.data)
                            l1_bias = layer.mlp.linear_fc2.bias.data

                    # concat mlp tensors
                    message["mlp l1 weight"] = torch.cat(l1_weight, dim=1)
                    if md.swiglu:
                        for tp_rank in range(tp_size):
                            l0_weight[tp_rank] = torch.chunk(l0_weight[tp_rank], 2, dim=0)
                        message["mlp l0 weight W"] = torch.cat([w[0] for w in l0_weight], dim=0)
                        message["mlp l0 weight V"] = torch.cat([w[1] for w in l0_weight], dim=0)
                    else:
                        message["mlp l0 weight"] = torch.cat(l0_weight, dim=0)

                    if md.add_bias_linear:
                        message["mlp l1 bias"] = l1_bias
                        if md.swiglu:
                            for tp_rank in range(tp_size):
                                l0_bias[tp_rank] = torch.chunk(l0_bias[tp_rank], 2, dim=0)
                            message["mlp l0 bias W"] = torch.cat([b[0] for b in l0_bias],dim=0)
                            message["mlp l0 bias V"] = torch.cat([b[1] for b in l0_bias],dim=0)
                        else:
                            message["mlp l0 bias"] = torch.cat(l0_bias, dim=0)

                queue_put(f"transformer layer {total_layer_num}", message)
                total_layer_num = total_layer_num + 1

    # Send final norm from tp_rank 0
    message = {"weight": models[0].decoder.final_layernorm.weight.data}
    if md.norm_has_bias:
        message["bias"] = models[0].decoder.final_layernorm.bias.data
    queue_put("final norm", message)

    if md.output_layer:
        output_layer_weight = []
        complete_tp_ranks = []
        for tp_ep_rank, model in enumerate(models):
            tp_rank = tp_ep_rank % tp_size
            if tp_rank in complete_tp_ranks:
                continue
            complete_tp_ranks.append(tp_rank)
            output_layer_weight.append(model.output_layer.weight.data)
        message = {"weight": torch.cat(output_layer_weight, dim=0)}
        queue_put("output layer", message)

    queue.put("done")

def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
