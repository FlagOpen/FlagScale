import megatron
import argparse
import dataclasses
import json
import os
import torch
import types

import torch.nn.functional as F
from megatron.global_vars import set_retro_args, get_retro_args
from megatron.core.transformer import TransformerConfig
from tools.retro.utils import get_args_path as get_retro_args_path
from megatron.arguments import _check_arg_is_not_none, _print_args

def validate_args(args, defaults={}):
    # Tensor model parallel size.
    args.tensor_model_parallel_size = min(
        args.tensor_model_parallel_size, args.world_size)
    assert args.world_size % args.tensor_model_parallel_size == 0, 'world size'\
        ' ({}) is not divisible by tensor model parallel size ({})'.format(
            args.world_size, args.tensor_model_parallel_size)
    # Pipeline model parallel size.
    args.pipeline_model_parallel_size = min(
        args.pipeline_model_parallel_size,
        (args.world_size // args.tensor_model_parallel_size))
    args.transformer_pipeline_model_parallel_size = (
        args.pipeline_model_parallel_size - 1
        if args.standalone_embedding_stage else
        args.pipeline_model_parallel_size
    )
    # Checks.
    model_parallel_size = args.pipeline_model_parallel_size * \
                          args.tensor_model_parallel_size
    assert args.world_size % model_parallel_size == 0, 'world size ({}) is not'\
        ' divisible by tensor parallel size ({}) times pipeline parallel ' \
        'size ({})'.format(args.world_size, args.tensor_model_parallel_size,
                           args.pipeline_model_parallel_size)
    args.data_parallel_size = args.world_size // model_parallel_size
    if args.rank == 0:
        print('using world size: {}, data-parallel-size: {}, '
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {} '.format(
                  args.world_size, args.data_parallel_size,
                  args.tensor_model_parallel_size,
                  args.pipeline_model_parallel_size), flush=True)
    if args.pipeline_model_parallel_size > 1:
        if args.pipeline_model_parallel_split_rank is not None:
            assert args.pipeline_model_parallel_split_rank < \
                    args.pipeline_model_parallel_size, 'split rank needs'\
                    ' to be less than pipeline model parallel size ({})'.format(
                            args.pipeline_model_parallel_size)

    # Deprecated arguments
    assert args.batch_size is None, '--batch-size argument is no longer ' \
        'valid, use --micro-batch-size instead'
    del args.batch_size
    assert args.warmup is None, '--warmup argument is no longer valid, use ' \
        '--lr-warmup-fraction instead'
    del args.warmup
    assert args.model_parallel_size is None, '--model-parallel-size is no ' \
        'longer valid, use --tensor-model-parallel-size instead'
    del args.model_parallel_size

    if args.checkpoint_activations:
        if args.rank == 0:
            print('--checkpoint-activations is no longer valid, use --recompute-activations, '
                  'or, for more control, --recompute-granularity and --recompute-method.')
        exit()
    del args.checkpoint_activations

    if args.recompute_activations:
        args.recompute_granularity = 'selective'
    del args.recompute_activations

    # rms layernorm
    if args.apply_layernorm_rms:
        args.apply_layernorm_1p = False

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0
    if args.num_layers_per_virtual_pipeline_stage is not None:
        assert args.pipeline_model_parallel_size > 2, \
            'pipeline-model-parallel size should be greater than 2 with ' \
            'interleaved schedule'
        assert args.num_layers % args.num_layers_per_virtual_pipeline_stage == 0, \
            'number of layers is not divisible by number of layers per virtual ' \
            'pipeline stage'
        args.virtual_pipeline_model_parallel_size = \
            (args.num_layers // args.transformer_pipeline_model_parallel_size) // \
            args.num_layers_per_virtual_pipeline_stage
    else:
        args.virtual_pipeline_model_parallel_size = None

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        assert not args.bf16
        args.params_dtype = torch.half
    if args.bf16:
        assert not args.fp16
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if not args.accumulate_allreduce_grads_in_fp32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    # If we do accumulation and all-reduces in fp32, we need to have local DDP
    # and we should make sure use-contiguous-buffers-in-local-ddp is not off.
    if args.accumulate_allreduce_grads_in_fp32:
        assert args.DDP_impl == 'local'
        assert args.use_contiguous_buffers_in_local_ddp

    # If we use the distributed optimizer, we need to have local DDP
    # and we should make sure use-contiguous-buffers-in-local-ddp is on.
    if args.use_distributed_optimizer:
        assert args.DDP_impl == 'local'
        assert args.use_contiguous_buffers_in_local_ddp

    # For torch DDP, we do not use contiguous buffer
    if args.DDP_impl == 'torch':
        args.use_contiguous_buffers_in_local_ddp = False

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0

    # Support for variable sequence lengths across batches/microbatches.
    # set it if the dataloader supports generation of variable sequence lengths
    # across batches/microbatches. Due to additional communication overhead
    # during pipeline parallelism, it should not be set if sequence length
    # is constant during training.
    args.variable_seq_lengths = False

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        assert args.train_samples is None, \
            'expected iteration-based training'
        assert args.lr_decay_samples is None, \
            'expected iteration-based learning rate decay'
        assert args.lr_warmup_samples == 0, \
            'expected iteration-based learning rate warmup'
        assert args.rampup_batch_size is None, \
            'expected no batch-size rampup for iteration-based training'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_iters == 0, \
                'can only specify one of lr-warmup-fraction and lr-warmup-iters'

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        assert args.train_iters is None, \
            'expected sample-based training'
        assert args.lr_decay_iters is None, \
            'expected sample-based learning rate decay'
        assert args.lr_warmup_iters == 0, \
            'expected sample-based learnig rate warmup'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_samples == 0, \
                'can only specify one of lr-warmup-fraction ' \
                'and lr-warmup-samples'

    if args.num_layers is not None:
        assert args.encoder_num_layers is None, \
            'cannot have both num-layers and encoder-num-layers specified'
        args.encoder_num_layers = args.num_layers
    else:
        assert args.encoder_num_layers is not None, \
            'either num-layers or encoder-num-layers should be specified'
        args.num_layers = args.encoder_num_layers

    # Check required arguments.
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads',
                     'max_position_embeddings']
    for req_arg in required_args:
        _check_arg_is_not_none(args, req_arg)

    # Checks.
    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = 4 * args.hidden_size

    if args.swiglu and args.mup is None:
        # Ref: https://github.com/facebookresearch/llama/blob/main/llama/model.py#L161-L162
        if args.multiple_of is not None:
            hidden_dim = int(4 * args.hidden_size * 2 / 3)
            if args.hidden_dim_multiplier is not None:
                assert args.hidden_dim_multiplier > 0, \
                    'multiplier for hidden dim should be greater than zero'
                hidden_dim = int(hidden_dim * args.hidden_dim_multiplier)
            args.ffn_hidden_size = args.multiple_of * \
                ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        else:
            # reduce the dimnesion for MLP since projections happens on
            # two linear layers. this keeps the number of paramters in
            # the same ballpark as the counterpart with 4*h size
            # we keep it a multiple of 64, which means the actual tensor size
            # will be a multiple of 64 / tp_size
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64

    if args.kv_channels is None:
        assert args.hidden_size % args.num_attention_heads == 0
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.seq_length is not None:
        assert args.encoder_seq_length is None
        args.encoder_seq_length = args.seq_length
    else:
        assert args.encoder_seq_length is not None
        args.seq_length = args.encoder_seq_length

    if args.seq_length is not None:
        assert args.max_position_embeddings >= args.seq_length
    if args.decoder_seq_length is not None:
        assert args.max_position_embeddings >= args.decoder_seq_length
    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.fp32_residual_connection:
        assert args.fp16 or args.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.'

    if args.weight_decay_incr_style == 'constant':
        assert args.start_weight_decay is None
        assert args.end_weight_decay is None
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    else:
        assert args.start_weight_decay is not None
        assert args.end_weight_decay is not None

    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    # Persistent fused layer norm.
    if TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 11):
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

    # Activation recomputing.
    if args.distribute_saved_activations:
        assert args.tensor_model_parallel_size > 1, 'can distribute ' \
            'recomputed activations only across tensor model ' \
            'parallel groups'
        assert args.recompute_granularity == 'full', \
            'distributed recompute activations is only '\
            'application to full recompute granularity'
        assert args.recompute_method is not None, \
            'for distributed recompute activations to work you '\
            'need to use a recompute method '
        assert (TORCH_MAJOR, TORCH_MINOR) >= (1, 10), \
            'distributed recompute activations are supported for pytorch ' \
            'v1.10 and above (Nvidia Pytorch container >= 21.07). Current ' \
            'pytorch version is v%s.%s.' % (TORCH_MAJOR, TORCH_MINOR)

    if args.recompute_granularity == 'selective':
        assert args.recompute_method is None, \
            'recompute method is not yet supported for ' \
            'selective recomputing granularity'

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.tensor_model_parallel_size == 1:
        args.sequence_parallel = False

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.sequence_parallel:
        args.async_tensor_model_parallel_allreduce = False

    if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == "1": # TODO (yehua.zhang): confirm MUSA_DEVICE_MAX_CONNECTIONS != 1
        if args.sequence_parallel:
            raise RuntimeError(
                "Using sequence parallelism requires setting the environment variable "
                "CUDA_DEVICE_MAX_CONNECTIONS to 1")
        if args.async_tensor_model_parallel_allreduce:
            raise RuntimeError(
                "Using async gradient all reduce requires setting the environment "
                "variable CUDA_DEVICE_MAX_CONNECTIONS to 1")

    # Disable bias gelu fusion if we are disabling bias altogether
    if not args.add_bias_linear:
        args.bias_gelu_fusion = False

    # Retro checks.
    if args.retro_add_retriever:

        # Sequence parallelism unsupported.
        assert not args.sequence_parallel, \
            "retro currently does not support sequence parallelism."

        # Pipeline parallelism unsupported.
        assert args.pipeline_model_parallel_size == 1, \
            "retro currently does not support pipeline parallelism."

        # Load retro args.
        retro_args_path = get_retro_args_path(args.retro_workdir)
        assert os.path.exists(retro_args_path), "retro workdir missing args.json"
        with open(retro_args_path) as f:
            retro_args = types.SimpleNamespace(**json.load(f))
            retro_args.retro_return_doc_ids = args.retro_return_doc_ids
            retro_args.retro_gpt_retrieved_length = \
                args.retro_num_retrieved_chunks * \
                retro_args.retro_gpt_chunk_length
            set_retro_args(retro_args)

    # Legacy RoPE arguments
    if args.use_rotary_position_embeddings:
        args.position_embedding_type = 'rope'

    # Would just need to add 'NoPE' as a position_embedding_type to support this, but for now
    # don't allow it to keep things simple
    if not args.add_position_embedding and args.position_embedding_type != 'rope':
        raise RuntimeError('--no-position-embedding is deprecated, use --position-embedding-type')

    # Print arguments.
    _print_args("arguments", args)
    retro_args = get_retro_args()
    if retro_args and args != retro_args:
        _print_args("retro arguments", types.SimpleNamespace(**{k:v for k,v in vars(retro_args).items() if k.startswith("retro")}, rank=args.rank))

    # Mup
    if args.mup is None:
        args.mup_coord_check = False

    # Check for scaled init_method
    if args.apply_init_customized:
        assert args.init_method_std_scaled_embed is not None
        assert args.num_layers == len(args.init_method_std_scaled_attn_q)
        assert args.num_layers == len(args.init_method_std_scaled_attn_k)
        assert args.num_layers == len(args.init_method_std_scaled_attn_v)
        assert args.num_layers == len(args.init_method_std_scaled_ffn_w1)
        assert args.num_layers == len(args.init_method_std_scaled_ffn_w2)
        assert args.num_layers == len(args.init_method_std_scaled_ffn_w3)
        assert args.init_method_std_scaled_output is not None

    # Check for norm init_weight
    if args.apply_init_norm_customized:
        assert args.num_layers == len(args.init_weight_attn_norm)
        assert args.num_layers == len(args.init_weight_ffn_norm)
        assert args.init_weight_output_norm is not None

    # Data Searching
    if args.data_searching_save:
        args.return_doc_ids = True
        assert args.world_size == 1, \
            "World size {args.world_size} should be one when data searching."

    return args

def core_transformer_config_from_args(args):

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(TransformerConfig):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype
    # FIXME (yehua.zhang) FIX batch_p2p_comm
    kw_args['batch_p2p_comm'] = False # not args.overlap_p2p_comm
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_gelu_fusion'] = False
    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_
    if args.group_query_attention:
        kw_args['num_query_groups'] = args.num_query_groups
    else:
        kw_args['num_query_groups'] = None

    return TransformerConfig(**kw_args)

def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--micro-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')
    group.add_argument('--batch-size', type=int, default=None,
                       help='Old batch size parameter, do not use. '
                       'Use --micro-batch-size instead')
    group.add_argument('--global-batch-size', type=int, default=None,
                       help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    group.add_argument('--rampup-batch-size', nargs='*', default=None,
                       help='Batch size ramp up with the following values:'
                       '  --rampup-batch-size <start batch size> '
                       '                      <batch size incerement> '
                       '                      <ramp-up samples> '
                       'For example:'
                       '   --rampup-batch-size 16 8 300000 \ '
                       '   --global-batch-size 1024'
                       'will start with global batch size 16 and over '
                       ' (1024 - 16) / 8 = 126 intervals will increase'
                       'the batch size linearly to 1024. In each interval'
                       'we will use approximately 300000 / 126 = 2380 samples.')
    group.add_argument('--recompute-activations', action='store_true',
                       help='recompute activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--recompute-granularity', type=str, default=None,
                       choices=['full', 'selective'],
                       help='Checkpoint activations to allow for training '
                       'with larger models, sequences, and batch sizes. '
                       'It is supported at two granularities 1) full: '
                       'whole transformer layer is recomputed, '
                       '2) selective: core attention part of the transformer '
                       'layer is recomputed.')
    group.add_argument('--distribute-saved-activations',
                       action='store_true',
                       help='If set, distribute recomputed activations '
                       'across model parallel group.')
    group.add_argument('--recompute-method', type=str, default=None,
                       choices=['uniform', 'block'],
                       help='1) uniform: uniformly divide the total number of '
                       'Transformer layers and recompute the input activation of '
                       'each divided chunk at specified granularity, '
                       '2) recompute the input activations of only a set number of '
                       'individual Transformer layers per pipeline stage and do the '
                       'rest without any recomputing at specified granularity'
                       'default) do not apply activations recompute to any layers')
    group.add_argument('--recompute-num-layers', type=int, default=None, # TODO (yehua.zhang) confirm recompute-num-layers
                       help='1) uniform: the number of Transformer layers in each '
                       'uniformly divided recompute unit, '
                       '2) block: the number of individual Transformer layers '
                       'to recompute within each pipeline stage.')
    group.add_argument('--profile', action='store_true',
                       help='Enable nsys profiling. When using this option, nsys '
                       'options should be specified in commandline. An example '
                       'nsys commandline is `nsys profile -s none -t nvtx,cuda '
                       '-o <path/to/output_file> --force-overwrite true '
                       '--capture-range=cudaProfilerApi '
                       '--capture-range-end=stop`.')
    group.add_argument('--profile-step-start', type=int, default=10,
                       help='Gloable step to start profiling.')
    group.add_argument('--profile-step-end', type=int, default=12,
                       help='Gloable step to stop profiling.')
    group.add_argument('--profile-ranks', nargs='+', type=int, default=[0],
                       help='Global ranks to profile.')


    # deprecated
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='Checkpoint activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--train-iters', type=int, default=None,
                       help='Total number of iterations to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--train-samples', type=int, default=None,
                       help='Total number of samples to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Report loss and timing interval.')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after the iteration is divisible '
                       'by this value.')
    group.add_argument('--exit-duration-in-mins', type=int, default=None,
                       help='Exit the program after this many minutes.')
    group.add_argument('--exit-signal-handler', action='store_true',
                       help='Dynamically save the checkpoint and shutdown the '
                       'training if SIGTERM is received')
    group.add_argument('--tensorboard-dir', type=str, default=None,
                       help='Write TensorBoard logs to this directory.')
    group.add_argument('--wandb-dir', type=str, default=None,
                       help='Write Wandb(Weights & Biases) logs to this directory.')
    group.add_argument('--no-masked-softmax-fusion',
                       action='store_false',
                       help='Disable fusion of query_key_value scaling, '
                       'masking, and softmax.',
                       dest='masked_softmax_fusion')
    group.add_argument('--no-bias-gelu-fusion', action='store_false',
                       help='Disable bias and gelu fusion.',
                       dest='bias_gelu_fusion')
    group.add_argument('--no-bias-dropout-fusion', action='store_false',
                       help='Disable bias and dropout fusion.',
                       dest='bias_dropout_fusion')
    group.add_argument('--use-flash-attn', action='store_true',
                       help='use FlashAttention implementation of attention, '
                       'FlashAttention v2 implementation as higher priority. '
                       'v1: https://arxiv.org/abs/2205.14135, '
                       'v2: https://tridao.me/publications/flash2/flash2.pdf')
    group.add_argument('--disable-bias-linear', action='store_false',
                       help='Disable bias in the linear layers',
                       dest='add_bias_linear')
    group.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'adan'],
                       help='Optimizer function')
    group.add_argument('--dataloader-type', type=str, default=None,
                       choices=['single', 'cyclic'],
                       help='Single pass vs multiple pass data loader')
    group.add_argument('--no-async-tensor-model-parallel-allreduce',
                       action='store_false',
                       help='Disable asynchronous execution of '
                       'tensor-model-parallel all-reduce with weight '
                       'gradient compuation of a column-linear layer.',
                       dest='async_tensor_model_parallel_allreduce')
    group.add_argument('--no-persist-layer-norm', action='store_true',
                       help='Disable using persistent fused layer norm kernel. '
                       'This kernel supports only a set of hidden sizes. Please '
                       'check persist_ln_hidden_sizes if your hidden '
                       'size is supported.')
    group.add_argument('--sequence-parallel', action='store_true',
                       help='Enable sequence parallel optimization.')
    group.add_argument('--no-gradient-accumulation-fusion',
                       action='store_false',
                       help='Disable fusing gradient accumulation to weight '
                       'gradient computation of linear layers',
                       dest='gradient_accumulation_fusion')
    return parser

def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--tensor-model-parallel-size', type=int, default=1,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--pipeline-model-parallel-split-rank',
                       type=int, default=None,
                       help='Rank where encoder and decoder should be split.')
    group.add_argument('--model-parallel-size', type=int, default=None,
                       help='Old model parallel argument, do not use. Use '
                       '--tensor-model-parallel-size instead.')
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--overlap-p2p-communication',
                       action='store_true',
                       help='overlap pipeline parallel communication with forward and backward chunks',
                       dest='overlap_p2p_comm')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo', 'mccl'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--distributed-timeout-minutes', type=int, default=10,
                       help='Timeout minutes for torch.distributed.')
    group.add_argument('--DDP-impl', default='local',
                       choices=['local', 'torch'],
                       help='which DistributedDataParallel implementation '
                       'to use.')
    group.add_argument('--no-contiguous-buffers-in-local-ddp',
                       action='store_false', help='If set, dont use '
                       'contiguous buffer in local DDP.',
                       dest='use_contiguous_buffers_in_local_ddp')
    group.add_argument('--no-scatter-gather-tensors-in-pipeline', action='store_false',
                       help='Use scatter/gather to optimize communication of tensors in pipeline',
                       dest='scatter_gather_tensors_in_pipeline')
    group.add_argument('--use-ring-exchange-p2p', action='store_true',
                       default=False, help='If set, use custom-built ring exchange '
                       'for p2p communications. Note that this option will require '
                       'a custom built image that support ring-exchange p2p.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    group.add_argument('--lazy-mpu-init', type=bool, required=False,
                       help='If set to True, initialize_megatron() '
                       'skips DDP initialization and returns function to '
                       'complete it instead.Also turns on '
                       '--use-cpu-initialization flag. This is for '
                       'external DDP manager.' )
    group.add_argument('--use-cpu-initialization', action='store_true',
                       default=None, help='If set, affine parallel weights '
                       'initialization uses CPU' )
    group.add_argument('--empty-unused-memory-level', default=0, type=int,
                       choices=[0, 1, 2],
                       help='Call torch.cuda.empty_cache() each iteration '
                       '(training and eval), to reduce fragmentation.'
                       '0=off, 1=moderate, 2=aggressive.')
    group.add_argument('--standalone-embedding-stage', action='store_true',
                       default=False, help='If set, *input* embedding layer '
                       'is placed on its own pipeline stage, without any '
                       'transformer layers. (For T5, this flag currently only '
                       'affects the encoder embedding.)')
    group.add_argument('--use-distributed-optimizer', action='store_true',
                       help='Use distributed optimizer.')
    group.add_argument('--no-global-file-system', action='store_true', 
                       default=False, help='If set, the trianing wonnot use the global file system.')
    group.add_argument('--num-devices-per-node', type=int, default=8,
                       help='Number of devices per node.')

    return parser

import sys
megatron.arguments.validate_args = validate_args
megatron.arguments.core_transformer_config_from_args = core_transformer_config_from_args
for k in sys.modules:
    if getattr(sys.modules[k], 'core_transformer_config_from_args', None):
        setattr(sys.modules[k], 'core_transformer_config_from_args', core_transformer_config_from_args)
    # if k.startswith('megatron'):
    #     if getattr(sys.modules[k], 'validate_args', None):
    #         setattr(sys.modules[k], 'validate_args', validate_args)
