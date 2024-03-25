import os
import sys
import importlib


import torch
from transformers import AutoModelForCausalLM


def add_arguments(parser):
    group = parser.add_argument_group(title='Transformers saver')

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


def save_checkpoint(queue, args):
    try:
        import transformers
        major, minor, _ = map(int, transformers.__version__.split('.'))
        assert major >= 4 and minor >= 31
    except:
        raise ImportError("transformers version >= 4.31.0 ")

    # Search in directory above this
    root_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir))
    sys.path.append(os.path.join(root_path, "megatron"))

    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.arguments import parse_args, validate_args
        from megatron.tokenizer.tokenizer import _vocab_size_with_padding
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    try:
        args_plugin = importlib.import_module(args.model_type + ".args")
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

    md = queue.get()

    assert args.target_tensor_parallel_size in [None, 1]
    assert args.target_pipeline_parallel_size in [None, 1]
    assert args.target_expert_parallel_size in [None, 1]
    # assert args.target_num_experts is None

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
    if args.target_num_experts is not None:
        sys.argv.extend(['--num-experts', str(args.target_num_experts)])
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

    print("*"*20 + "validate saver arguments" + "*"*20)
    margs = validate_args(margs)

    def padding_vocab_size(orig_word_embed):
        if md.true_vocab_size is not None:
            # figure out what our padded vocab size is
            orig_vocab_size = orig_word_embed.shape[0]
            margs.padded_vocab_size = _vocab_size_with_padding(md.true_vocab_size, margs)

            # Cut out extra padding we don't need
            if orig_vocab_size > margs.padded_vocab_size:
                full_word_embed = orig_word_embed[0:margs.padded_vocab_size, :]

            # Expanding embedding to larger size by replicating final entry
            elif orig_vocab_size < margs.padded_vocab_size:
                padding_size = margs.padded_vocab_size - orig_vocab_size

                full_word_embed = torch.cat((
                    orig_word_embed,
                    orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1)
                ))

            # Same size!
            else:
                full_word_embed = orig_word_embed
        else:
            print("Original vocab size not specified, leaving embedding table as-is. "
                "If you've changed the tensor parallel size this could cause problems.")
            margs.padded_vocab_size = orig_word_embed.shape[0]
            full_word_embed = orig_word_embed
        return full_word_embed

    embeddings_msg = queue_get("embeddings")
    origin_embed = embeddings_msg.pop("word embeddings")
    full_word_embed = padding_vocab_size(origin_embed)

    hf_config = args_plugin.save_args_mg2hf(margs)
    hf_model = model_plugin.get_hf_model(margs.params_dtype, config=hf_config)

    hf_model.model.embed_tokens.weight.data.copy_(full_word_embed)
    check_message(embeddings_msg)

    for layer_id in range(md.num_layers):
        msg = queue_get(f"transformer layer {layer_id}")

        ckpt_plugin.set_hf_attn_ckpt(msg, hf_model, layer_id, md, margs)
        ckpt_plugin.set_hf_mlp_ckpt(msg, hf_model, layer_id, md, margs)

        check_message(msg)

    msg = queue_get("final norm")
    hf_model.model.norm.weight.data.copy_(msg.pop("weight"))
    if md.norm_has_bias:
        hf_model.model.norm.bias.data.copy_(msg.pop("bias"))
    check_message(msg)

    if md.output_layer:
        msg = queue_get("output layer")
        orig_output_layer_weight = msg.pop("weight")
        full_output_layer_weight = padding_vocab_size(orig_output_layer_weight)
        hf_model.lm_head.weight.data.copy_(full_output_layer_weight)
        check_message(msg)

    hf_model.save_pretrained(args.save_dir)
