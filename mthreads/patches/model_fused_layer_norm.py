import torch
import torch_musa
import megatron

class MixedFusedLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6,
                 no_persist_layer_norm=True,
                 sequence_parallel=False,
                 apply_layernorm_1p=False,
                 apply_layernorm_rms=False,
                 init_weight=None):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, device='musa:{}'.format(torch.musa.current_device())))
        self.variance_epsilon = eps
        self.sequence_parallel = sequence_parallel
        self.apply_layernorm_rms = apply_layernorm_rms
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
        if not self.apply_layernorm_rms:
            setattr(self.bias, 'sequence_parallel', self.sequence_parallel)

    def forward(self, hidden_states):

        return torch.rms_norm(hidden_states, (hidden_states.size(-1),), self.weight, self.variance_epsilon)
    
import sys
for k in sys.modules:
    if k.startswith('megatron.model'):
        for target in ['LayerNorm', 'MixedFusedLayerNorm']:
            if getattr(sys.modules[k], target, None):
                setattr(sys.modules[k], target, MixedFusedLayerNorm)
