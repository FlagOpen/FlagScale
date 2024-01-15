import megatron
import dataclasses
import torch

import torch.nn.functional as F
from megatron.core.transformer import TransformerConfig

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

import sys
for k in sys.modules:
    if getattr(sys.modules[k], 'core_transformer_config_from_args', None):
        setattr(sys.modules[k], 'core_transformer_config_from_args', core_transformer_config_from_args)
