import megatron
import torch
from .core_transformer_custom_layers import Norm
from megatron.core.transformer.transformer_block import build_module
from flagscale.patches_utils import add_patches_module


def _build_layers(self):
        # Transformer layers.
        # @jcasper can we improve how we deal with layer_number?
        # currently it's only used in CoreAttention?
        def build_layer(layer_spec, layer_number):
            return build_module(layer_spec, config=self.config, layer_number=layer_number,)

        # offset is implicit in TransformerLayer
        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

       
        if self.post_process and self.post_layer_norm:
            # [metax] start of change
            self.final_layernorm = Norm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
            # [metax] end of change


func_path = "megatron.core.transformer.transformer_block"
func_dict = {"TransformerBlock._build_layers":_build_layers}
add_patches_module(func_path, func_dict)
