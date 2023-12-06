# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

try:
    from .fused_layer_norm import MixedFusedLayerNorm as LayerNorm
    from .rms_norm import RMSNorm
except Exception as e:
    # from torch.nn import LayerNorm
    print('WARNING: APEX is not installed and is not supported in KL yet')
    from torch.nn import LayerNorm as TorchLayerNorm
    class LayerNorm(TorchLayerNorm):
        """Inherit from torch.nn.LayerNorm but eliminate extra kwargs"""
        def __init__(self, normalized_shape, eps=1e-5,
                    no_persist_layer_norm=True,
                    sequence_parallel=False,
                    apply_layernorm_1p=False):
                super().__init__(
                    normalized_shape, eps = eps)
                self.sequence_parallel = sequence_parallel
                setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
                setattr(self.bias, 'sequence_parallel', self.sequence_parallel)

from .bert_model import BertModel
from .gpt_model import GPTModel
from .t5_model import T5Model
from .language_model import get_language_model
from .module import Float16Module
