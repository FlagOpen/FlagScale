import torch
import megatron
from megatron.model.module import conversion_helper


def fp32_to_float16(val, float16_convertor):
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (torch.nn.parameter.Parameter, torch.autograd.Variable)):
            val_typecheck = val.data
        if val_typecheck.dtype == torch.float32:
            val = float16_convertor(val)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (torch.nn.parameter.Parameter, torch.autograd.Variable)):
            val_typecheck = val.data
        if val_typecheck.dtype in [torch.float16, torch.bfloat16]:
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


megatron.model.module.fp32_to_float16 = fp32_to_float16
megatron.model.module.float16_to_fp32 = float16_to_fp32
