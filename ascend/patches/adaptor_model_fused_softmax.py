import torch_npu
import megatron


def is_kernel_available(self, mask, b, np, sq, sk):
    return (
            self.scaled_masked_softmax_fusion  # user want to fuse
            and self.input_in_float16  # input must be fp16
            and 32 < sk <= 2048  # sk must be 32 ~ 2048
            and sq % 16 == 0  # sq must be divisor of 16
            and sk % 16 == 0  # sk must be divisor of 16
    )


def forward_fused_softmax(self, input, mask):
    return torch_npu.npu_scaled_masked_softmax(input, mask, self.scale, False)


megatron.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available = is_kernel_available
megatron.model.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax = forward_fused_softmax
