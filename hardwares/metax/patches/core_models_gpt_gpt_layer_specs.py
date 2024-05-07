
import megatron
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from .core_transformer_custom_layers import Norm
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from flagscale.patches_utils import add_patches_func_

# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_gpt_layer_with_transformer_engine_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    # [metax] start of change
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    # [metax] end of change
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=Norm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    # [metax] start of change
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=Norm if qk_layernorm else IdentityOp,
                    k_layernorm=Norm if qk_layernorm else IdentityOp,
                    # [metax] end of change
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            
            # [metax] start of change
            pre_mlp_layernorm=Norm if num_experts else IdentityOp,
            # [metax] end of change
            
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            
            # [metax] start of change
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
            # [metax] end of change
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_gpt_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            # [metax] start of change
            input_layernorm=Norm,
            # [metax] end of change
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,

                    # [metax] start of change
                    q_layernorm=Norm if qk_layernorm else IdentityOp,
                    k_layernorm=Norm if qk_layernorm else IdentityOp,
                    # [metax] end of change
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            # [metax] start of change
            pre_mlp_layernorm=Norm,
            # [metax] end of change
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(
    use_te: bool = True, num_experts: int = None, moe_grouped_gemm: bool = False
) -> ModuleSpec:
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(

                # [metax] start of change
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
                # [metax] end of change
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return ModuleSpec(
            module=MoELayer,
            submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,)
            if not moe_grouped_gemm
            else None,
        )


func_path = "megatron.core.models.gpt.gpt_layer_specs"
func_dict = {"get_gpt_layer_with_transformer_engine_spec":get_gpt_layer_with_transformer_engine_spec,
             "get_gpt_layer_local_spec":get_gpt_layer_local_spec,
             "_get_mlp_module_spec":_get_mlp_module_spec}
add_patches_func_(func_path,func_dict)

# megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec = get_gpt_layer_with_transformer_engine_spec
# megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec = get_gpt_layer_local_spec
# megatron.core.models.gpt.gpt_layer_specs._get_mlp_module_spec = _get_mlp_module_spec

