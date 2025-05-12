# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import torch


def _make_param_hook(
        self,
        param: torch.nn.Parameter,
        param_to_buffer,
):
    """
    Creates the all-reduce / reduce-scatter hook for backprop.
    """

    def param_hook(*unused):
        if param.requires_grad and not getattr(param, 'skip_grad_accum', False):
            if self.ddp_config.overlap_grad_reduce:
                assert (
                        param.grad is not None
                ), 'param.grad being None is not safe when overlap_grad_reduce is True'
            if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
            ):
                param.main_grad.add_(param.grad.data)
            param.grad = None

        # Maybe should called after weightgradstore.pop()
        if self.ddp_config.overlap_grad_reduce:
            param_to_buffer[param].register_grad_ready(param)
        if getattr(param, 'skip_grad_accum', False):
            param.skip_grad_accum = False

    return param_hook