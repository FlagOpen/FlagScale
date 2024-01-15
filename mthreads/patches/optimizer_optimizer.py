import torch
import megatron
from megatron.core import mpu, tensor_parallel
from megatron.optimizer.optimizer import MixedPrecisionOptimizer, Float16OptimizerWithFloat16Params,FP32Optimizer

def MixedPrecisionOptimizer_init(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 fp16, bf16, params_dtype, grad_scaler,
                 models):

    super(MixedPrecisionOptimizer, self).__init__(
        optimizer, clip_grad, log_num_zeros_in_grad,
        params_have_main_grad, use_contiguous_buffers_in_local_ddp,
        models)

    self.fp16 = fp16
    self.bf16 = bf16
    self.params_dtype = params_dtype
    self.grad_scaler = grad_scaler

    # None grad scaler is only supported for bf16.
    if self.grad_scaler is None:
        assert not self.fp16, 'fp16 expects a grad scaler.'

    # Tensor used to determine if a nan/if has happend.
    # Any non-zero value indicates inf/nan.
    # Note that we keep this for the cases that grad scaler is none.
    # We still record nan/inf if we have a bfloat16 with a grad scaler.
    if self.grad_scaler:
        self.found_inf = torch.musa.FloatTensor([0.0])

    # Dummy tensor needed for apex multi-apply tensor.
    # For bfloat, we don't have multi-tensor apply and for now
    # we set it to none so the multi-tensor apply gets ignored.
    if bf16:
        self._dummy_overflow_buf = None
    else:
        self._dummy_overflow_buf = torch.musa.IntTensor([0])

    # In case grad scaler is not passed, define the unity scale.
    if self.grad_scaler is None:
            self._scale_one = torch.musa.FloatTensor([1.0])


def Float16OptimizerWithFloat16Params_init(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 fp16, bf16, params_dtype, grad_scaler, models):

    super(Float16OptimizerWithFloat16Params, self).__init__(
        optimizer, clip_grad, log_num_zeros_in_grad,
        params_have_main_grad, use_contiguous_buffers_in_local_ddp,
        fp16, bf16, params_dtype, grad_scaler, models)
    print("Float16OptimizerWithFloat16Params init self define ##################")

    # ======================
    # main parameter stuff
    # ======================

    # Three groups of parameters:
    #   float16_groups: original float16 parameters
    #   fp32_from_float16_groups: fp32 copy of float16 parameters
    #   fp32_from_fp32_groups: original fp32 parameters
    self.float16_groups = []
    self.fp32_from_float16_groups = []
    self.fp32_from_fp32_groups = []

    # For all the groups in the original optimizer:
    for param_group in self.optimizer.param_groups:
        float16_params_this_group = []
        fp32_params_this_group = []
        fp32_from_float16_params_this_group = []
        # For all the parameters in this group:
        for i, param in enumerate(param_group['params']):
            if param.requires_grad:

                # float16 params:
                if param.type() in ['torch.musa.HalfTensor',
                                    'torch.musa.BFloat16Tensor']:
                    float16_params_this_group.append(param)
                    # Create a copy
                    main_param = param.detach().clone().float()
                    # Copy tensor model parallel attributes.
                    tensor_parallel.copy_tensor_model_parallel_attributes(main_param,
                                                                            param)
                    if hasattr(param, 'shared'):
                        main_param.shared = param.shared
                    # Replace the optimizer params with the new fp32 copy.
                    param_group['params'][i] = main_param

                    fp32_from_float16_params_this_group.append(main_param)
                    # Reset existing state dict key to the new main param.
                    if param in self.optimizer.state:
                        self.optimizer.state[main_param] \
                            = self.optimizer.state.pop(param)
                # fp32 params.
                elif param.type() == 'torch.musa.FloatTensor':
                    fp32_params_this_group.append(param)
                    param_group['params'][i] = param

                else:
                    raise TypeError('Wrapped parameters must be one of '
                                    'torch.musa.FloatTensor,  '
                                    'torch.musa.HalfTensor, or '
                                    'torch.musa.BFloat16Tensor. '
                                    'Received {}'.format(param.type()))

        self.float16_groups.append(float16_params_this_group)
        self.fp32_from_float16_groups.append(
            fp32_from_float16_params_this_group)
        self.fp32_from_fp32_groups.append(fp32_params_this_group)


def FP32Optimizer_init(self, optimizer, clip_grad,
                 log_num_zeros_in_grad,
                 params_have_main_grad,
                 use_contiguous_buffers_in_local_ddp,
                 models):

        super(FP32Optimizer, self).__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            models)

        self._scale = torch.musa.FloatTensor([1.0])


# import sys
# for k in sys.modules:
#     if getattr(sys.modules[k], 'Float16OptimizerWithFloat16Params', None):
#         print(k)
megatron.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__ = Float16OptimizerWithFloat16Params_init
megatron.optimizer.Float16OptimizerWithFloat16Params.__init__ = Float16OptimizerWithFloat16Params_init
# megatron.optimizer.optimizer.MixedPrecisionOptimizer.__init__ = MixedPrecisionOptimizer_init
# megatron.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__ = Float16OptimizerWithFloat16Params_init
# megatron.optimizer.optimizer.FP32Optimizer.__init__ = FP32Optimizer_init
