import os
import sys
import importlib

import torch


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-pipeline-parallel-size', type=int,
                       help='Target tensor model parallel size, default to the pipeline parall size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument("--target-expert-parallel-size", type=int,
                      help='Target expert model parallel size, default to the expert parallel size '
                      'in the input checkpoint if provided by the loader, otherwise to 1.')
    group.add_argument("--target-num-experts", type=int, default=None,
                       help='Target num of experts, default to the num_experts in the input checkpoint'
                       'if provided by the loader, otherwise to None. NOTE: Do not support target_num_experts'
                       'is not None and the num_experts is not equal to the provieded by the loader.')
    group.add_argument("--target-params-dtype", type=str, default=None,
                       help='The dtype of the converted checkpoint. '
                            'Only used when converting a Transformers checkpoint to a Megatron checkpoint.')
    group.add_argument("--build-model-with-initialization", action="store_true")


def save_checkpoint(queue, args):

    """
    prepare import module
    """

    # Search in directory above this
    root_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir))
    sys.path.append(os.path.join(root_path, "megatron"))
    sys.path.append(root_path)

    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.checkpointing import save_checkpoint, get_checkpoint_name
        from megatron.training.global_vars import set_global_variables
        from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
        from megatron.legacy import fused_kernels
        from megatron.core import mpu
        from megatron.core.tensor_parallel.random import (
                get_cuda_rng_tracker, _DATA_PARALLEL_RNG_TRACKER_NAME,
                _EXPERT_PARALLEL_RNG_TRACKER_NAME, _MODEL_PARALLEL_RNG_TRACKER_NAME
            )
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    try:
        ckpt_plugin = importlib.import_module(args.model_type + ".ckpt")
        model_plugin = importlib.import_module(args.model_type + ".model")
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Please check model_type or model.py")

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

    """
    prepare megatron arguments (margs)
    """

    md = queue_get()

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
            print("loader did not provide a expert parallel size and --target-expert-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_expert_parallel_size = 1

    if args.target_num_experts is None:
        if hasattr(md, 'previous_num_experts'):
            args.target_num_experts = md.previous_num_experts
        else:
            print("loader did not provide a num experts and --target-num-experts not provided on command line. "
                  "Default to None.")

    if args.target_num_experts is not None and md.previous_num_experts is not None:
        assert args.target_num_experts >= md.previous_num_experts, \
            "target_num_experts should be greater than previous_num_experts"
        if args.target_num_experts > md.previous_num_experts:
            print(f"Warning: experts[{md.previous_num_experts}-{args.target_num_experts}] will be random initailized.")

    os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size * args.target_pipeline_parallel_size * args.target_expert_parallel_size}'
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

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
        '--use-mcore-models',
        '--transformer-impl', 'transformer_engine',
        '--micro-batch-size', '1',
        '--no-load-optim',
        '--no-load-rng',
        '--no-save-optim',
        '--no-save-rng',
        '--save-interval', '1',
        '--save', args.save_dir
    ]
    if args.target_num_experts is not None:
        sys.argv.extend(['--num-experts', str(args.target_num_experts)])
        sys.argv.append('--sequence-parallel')
    if not args.build_model_with_initialization:
        sys.argv.append('--no-initialization')
    if md.make_vocab_size_divisible_by is not None:
        sys.argv.extend(['--make-vocab-size-divisible-by', str(md.make_vocab_size_divisible_by)])
    if md.output_layer:
        sys.argv.append('--untie-embeddings-and-output-weights')
    if not md.add_bias_linear:
        sys.argv.append('--disable-bias-linear')
    if md.add_qkv_bias:
        sys.argv.append('--add-qkv-bias')

    if args.target_params_dtype is not None:
        assert args.target_params_dtype in ["fp32", "fp16", "bf16"]
        if args.target_params_dtype == "fp32":
            md.params_dtype = torch.float32
        elif args.target_params_dtype == "fp16":
            md.params_dtype = torch.float16
        elif args.target_params_dtype == "bf16":
            md.params_dtype = torch.bfloat16
        print(f"> convert params_dtype to {md.params_dtype}")

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
                        'save_interval', 'save', 'load', 'use_mcore_models', 'num_experts',
                        'perform_initialization', 'use_cpu_initialization',
                        'recompute_granularity', 'recompute_num_layers', 'recompute_method',
                        'encoder_num_layers', 'encoder_seq_length',
                        'distribute_saved_activations', 'fp16', 'bf16',
                        'train_iters', 'lr_decay_iters', 'lr_warmup_iters', 'lr_warmup_fraction',
                        'start_weight_decay', 'end_weight_decay',
                        'main_grads_dtype', 'main_params_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype'
        ]

        for arg, value in vars(md.checkpoint_args).items():
            if arg in args_to_keep:
                continue
            if not hasattr(margs, arg):
                print(f"Checkpoint had argument {arg} but new arguments does not have this.")
                continue
            if getattr(margs, arg) != value:
                print(f"Overwriting default {arg} value {getattr(margs, arg)} with value from checkpoint {value}.")
                setattr(margs, arg, value)

    print("*"*20 + "validate saver arguments" + "*"*20)
    margs = validate_args(margs)

    # validate consumed_samples
    if hasattr(md, 'consumed_train_samples'):
        margs.consumed_train_samples = md.consumed_train_samples
        margs.consumed_valid_samples = md.consumed_valid_samples
        print(f"Setting consumed_train_samples to {margs.consumed_train_samples}"
              f" and consumed_valid_samples to {margs.consumed_valid_samples}")
    else:
        print("consumed_train_samples not provided.")

    # Determine how to make our models.
    margs.model_type = model_plugin.model_type

    if md.true_vocab_size is not None:
        margs.padded_vocab_size = _vocab_size_with_padding(md.true_vocab_size, margs)
    else:
        # margs.padded_vocab_size will be set in ckpt_plugin.set_embedding_ckpt func
        margs.padded_vocab_size = None

    """
    use megatron args build object and init env
    """

    # build global variable (eg: tokenizer)
    set_global_variables(margs, build_tokenizer=False)

    # fake initializing distributed
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    ep_size = margs.expert_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is not None and vp_size > 1:
        raise NotImplementedError("vpp-convert is not implemented")
    mpu.set_tensor_model_parallel_world_size(tp_size)
    mpu.set_pipeline_model_parallel_world_size(pp_size)
    mpu.set_expert_model_parallel_world_size(ep_size)
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)

    # fused kernel
    fused_kernels.load(margs)

    # random
    CUDA_RNG_STATE_TRACKER = get_cuda_rng_tracker()
    torch.cuda.manual_seed(42)
    CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, 43)
    CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 44)
    CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, 45)

    def get_models(count, dtype):
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        models = [model_plugin.get_mg_model(dtype, pre_process, post_process) for _ in range(count)]
        return models

    """
    start receive and process ckpt
    """

    # process embedding
    msg = queue_get("embeddings")
    mpu.set_pipeline_model_parallel_rank(0)
    models = get_models(tp_size * ep_size, md.params_dtype)
    ckpt_plugin.set_embedding_ckpt(msg, models, md, margs)
    check_message(msg)

    # process transformer layer
    total_layer_num = 0
    for pp_rank in range(pp_size):

        mpu.set_pipeline_model_parallel_rank(pp_rank)

        if pp_rank > 0:
            models = get_models(tp_size * ep_size, md.params_dtype)

        for layer_id in range(len(models[0].decoder.layers)):
            msg = queue_get(f"transformer layer {total_layer_num}")
            margs.total_layer_num = total_layer_num
            ckpt_plugin.set_attn_ckpt(msg, models, layer_id, md, margs)
            ckpt_plugin.set_mlp_ckpt(msg, models, layer_id, md, margs)

            total_layer_num = total_layer_num + 1
            check_message(msg)

        # process final layernorm and linear
        if pp_rank == pp_size - 1:
            msg = queue_get("final norm")
            ckpt_plugin.set_final_norm_ckpt(msg, models, md, margs)
            check_message(msg)

            if md.output_layer:
                msg = queue_get("output layer")
                assert hasattr(models[0], 'output_layer'), "ERROR: got an output layer, but model does not have one"
                ckpt_plugin.set_output_layer_ckpt(msg, models, md, margs)

            if margs.num_mtp_predictor > 0:
                for mtp_layer_id in range(margs.num_mtp_predictor):
                    msg = queue_get(f"mtp module {mtp_layer_id}")
                    ckpt_plugin.set_mtp_ckpt(msg, models, md, mtp_layer_id, margs)

            msg = queue_get()
            if msg != "done":
                print("ERROR: got some more data but was expecting to be done")
        margs.use_dist_ckpt = False
        for tp_ep_rank, model in enumerate(models):
            tp_rank = tp_ep_rank % tp_size
            ep_rank = tp_ep_rank // tp_size
            mpu.set_tensor_model_parallel_rank(tp_rank)
            mpu.set_expert_model_parallel_rank(ep_rank)
            checkpoint_name = get_checkpoint_name(margs.save, md.iteration)
            print(f"megtron model is saving to {checkpoint_name} ...")
            save_checkpoint(md.iteration, [model], None, None,
                            num_floating_point_operations_so_far=0)

    print("SAVE DONE!!!")
