import torch

from tests.unit_tests.test_utilities import Utils as MegatronUtils

from megatron.training.arguments import parse_args
import megatron.training.global_vars as mcore_global_vars
from megatron.training.tokenizer.tokenizer import _NullTokenizer


from flagscale.train.hetero.parallel_context import ParallelContext
from flagscale.train.arguments import FSTrainArguments # noqa


def init_parallel_context() -> ParallelContext:
    
    args = parse_args(ignore_unknown_args=True)
    args.tensor_model_parallel_size = 2
    args.pipeline_model_parallel_size = 3
    args.disable_bias_linear = True
    args.use_flash_attn = True
    args.sequence_parallel = True
    args.use_distributed_optimizer = True
    args.use_mcore_models = True
    args.transformer_impl = "transformer_engine"
    args.enable_hetero = True
    args.hetero_pipeline_layer_split = [6, 2, 4]
    args.hetero_process_meshes = [[2, 1, 1, 1, 1], [4, 1, 1, 1, 1], [2, 1, 1, 1, 1]]
    args.hetero_device_types = ["A800", "A800", "A800"]
    args.hetero_current_device_type = "A800"
    
    # extra transformer config
    args.params_dtype = torch.bfloat16
    args.num_attention_heads = 32
    args.hidden_size = 1024
    args.num_layers = 24
    
    # for building datasets
    mcore_global_vars._GLOBAL_TOKENIZER = _NullTokenizer(vocab_size=64)
    para_ctx = ParallelContext(args)
    return para_ctx


def test_parallel_config():
    MegatronUtils.initialize_distributed()
    
    para_ctx = init_parallel_context()
    
    assert para_ctx is not None
    assert para_ctx.get_ddp_config() is not None
    assert para_ctx.get_transformer_config() is not None
    assert para_ctx.get_dataset_config() is not None

