import torch
import megatron

_FLOAT_TYPES = (torch.FloatTensor, torch.musa.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.musa.HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor, torch.musa.BFloat16Tensor)

megatron.model.module._FLOAT_TYPES = _FLOAT_TYPES
megatron.model.module._HALF_TYPES = _HALF_TYPES
megatron.model.module._BF16_TYPES = _BF16_TYPES