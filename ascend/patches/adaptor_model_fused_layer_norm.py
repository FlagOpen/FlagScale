import torch
import torch_npu
import numbers
import megatron
from megatron.model.fused_layer_norm import MixedFusedLayerNorm
from megatron.core.utils import make_viewless_tensor

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN
    HAVE_PERSIST_LAYER_NORM = True
except:
    HAVE_PERSIST_LAYER_NORM = False


def MixedFusedLayerNormInit(self, normalized_shape, eps=1e-5,
                            no_persist_layer_norm=True,
                            sequence_parallel=False,
                            apply_layernorm_1p=False,
                            apply_layernorm_rms=False,
                            init_weight=None,
                            ):
    super(MixedFusedLayerNorm, self).__init__()

    self.apply_layernorm_1p = apply_layernorm_1p
    self.apply_layernorm_rms = apply_layernorm_rms
    assert not (self.apply_layernorm_1p and self.apply_layernorm_rms), \
        "Cannot apply both 1p and rms layernorm"
    
    self.init_weight = init_weight
    assert self.init_weight is None or isinstance(self.init_weight, float), \
        "Cannot init_weight of None or of non-float"
    assert not (self.init_weight is not None and self.apply_layernorm_1p), \
        "Cannot float init_weight and 1p layernorm"
    
    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape,)
    self.normalized_shape = torch.Size(normalized_shape)
    self.eps = eps
    self.weight = torch.nn.parameter.Parameter(torch.Tensor(*normalized_shape))
    # no bias parameter when using rms layernorm
    if not self.apply_layernorm_rms:
        self.bias = torch.nn.parameter.Parameter(torch.Tensor(*normalized_shape))
    self.reset_parameters()
    self.no_persist_layer_norm = True
    self.sequence_parallel = sequence_parallel

    # set sequence parallelism flag on weight and bias parameters
    setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
    if not self.apply_layernorm_rms:
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)


def MixedFusedLayerNormForward(self, input):
    weight = self.weight + 1 if self.apply_layernorm_1p else self.weight
    if self.apply_layernorm_rms:
        return torch_npu.npu_rms_norm(input, weight, epsilon=self.eps)[0]
    elif self.no_persist_layer_norm:
        return torch.nn.functional.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
    else:
        output = FastLayerNormFN.apply(input, self.weight, self.bias, self.eps)
        output = make_viewless_tensor(inp=output, requires_grad=input.requires_grad, keep_graph=True)
    return output


megatron.model.fused_layer_norm.MixedFusedLayerNorm.__init__ = MixedFusedLayerNormInit
megatron.model.fused_layer_norm.MixedFusedLayerNorm.forward = MixedFusedLayerNormForward
