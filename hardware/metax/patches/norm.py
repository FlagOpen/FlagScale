import megatron
from megatron.training import get_args
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.legacy.model import LayerNorm, RMSNorm



class Norm:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `LayerNorm` or `RMSNorm` based on input
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(
        cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5,
    ):
        args = get_args()
        if config.normalization == "LayerNorm":

            instance = LayerNorm(
                config.hidden_size,
                eps=config.layernorm_epsilon,
                no_persist_layer_norm=not config.persist_layer_norm,
                sequence_parallel=config.sequence_parallel,
                apply_layernorm_1p=args.apply_layernorm_1p)

        elif config.normalization == "RMSNorm":

            instance = RMSNorm(
                config.hidden_size,
                eps=config.layernorm_epsilon,
                no_persist_layer_norm=not config.persist_layer_norm,
                sequence_parallel=config.sequence_parallel,
                apply_layernorm_1p=args.apply_layernorm_1p)

        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance



import sys
for k in sys.modules:
    if k.startswith('megatron.core.transformer.custom_layers'):
        if getattr(sys.modules[k], 'Norm', None):
            setattr(sys.modules[k], 'Norm', Norm)
        
