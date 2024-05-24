import megatron
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
# [metax] start of changes
from .core_transformer_custom_layers import Norm
# [metax] end of changes
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from flagscale.patches_utils import add_patches_func

# Use this spec for AMMO PTQ and TensorRT-LLM export
def get_gpt_layer_ammo_spec() -> ModuleSpec:
    """Mix the native spec with TENorm.

    This is essentially the native local spec except for the layernorm implementation
    is using TENorm from Transformer-Engine. This TENorm supports both FusedLayerNorm and RMSNorm and
    prevents the apex dependency.
    """
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            # [metax] start of changes
            input_layernorm=Norm,
            # [metax] end of changes
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            # [metax] start of changes
            pre_mlp_layernorm=Norm,
            # [metax] end of changes
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            # Map TE-layernorm-fusion keys back
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


func_path = "megatron.core.inference.gpt.model_specs"
func_dict = {"get_gpt_layer_ammo_spec",get_gpt_layer_ammo_spec}
add_patches_func(func_path,func_dict)

