import sys
import torch
import torch_mlu
import megatron.optimizer
from megatron.optimizer.optimizer import MixedPrecisionOptimizer 

def MixedPrecisionOptimizerInit(self, optimizer, clip_grad, log_num_zeros_in_grad,
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
        self.found_inf = torch.mlu.FloatTensor([0.0])

    # Dummy tensor needed for apex multi-apply tensor.
    # For bfloat, we don't have multi-tensor apply and for now
    # we set it to none so the multi-tensor apply gets ignored.
    if bf16:
        self._dummy_overflow_buf = None
    else:
        self._dummy_overflow_buf = torch.mlu.IntTensor([0])

    # In case grad scaler is not passed, define the unity scale.
    if self.grad_scaler is None:
        self._scale_one = torch.mlu.FloatTensor([1.0])


def _unscale_main_grads_and_check_for_nan(self):
    # Collect main grads.
    main_grads = self._collect_main_grad_data_for_unscaling()

    # Reset found inf.
    self.found_inf.fill_(0.0)

    # Unscale and set found inf/nan
    # torch._amp_foreach_non_finite_check_and_unscale_(
    #     main_grads, self.found_inf, self.grad_scaler.inv_scale)
    output = torch.ops.torch_mlu.amp_unscale(main_grads, self.found_inf,
        self.grad_scaler.inv_scale)


    # Update across all model parallel instances.
    torch.distributed.all_reduce(self.found_inf,
                                 op=torch.distributed.ReduceOp.MAX,
                                 group=self.get_model_parallel_group())

    # Check for nan.
    found_inf_flag = (self.found_inf.item() > 0)

    return found_inf_flag

def Float16OptimizerWithFloat16ParamsInit(self, optimizer, clip_grad, log_num_zeros_in_grad,
             params_have_main_grad, use_contiguous_buffers_in_local_ddp,
             fp16, bf16, params_dtype, grad_scaler, models):

    super().__init__(
        optimizer, clip_grad, log_num_zeros_in_grad,
        params_have_main_grad, use_contiguous_buffers_in_local_ddp,
        fp16, bf16, params_dtype, grad_scaler, models)
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
                if param.type() in ['torch.mlu.HalfTensor',
                                    'torch.mlu.BFloat16Tensor',
                                    'torch.cuda.HalfTensor',
                                    'torch.cuda.BFloat16Tensor']:
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
                elif param.type() in ['torch.mlu.FloatTensor',
                                      'torch.cuda.FloatTensor']:
                    fp32_params_this_group.append(param)
                    param_group['params'][i] = param

                else:
                    raise TypeError('Wrapped parameters must be one of '
                                    'torch.mlu.FloatTensor, or '
                                    'torch.mlu.HalfTensor, or '
                                    'torch.mlu.BFloat16Tensor, or '
                                    'torch.cuda.FloatTensor, or '
                                    'torch.cuda.HalfTensor, or '
                                    'torch.cuda.BFloat16Tensor. '
                                    'Received {}'.format(param.type()))

        self.float16_groups.append(float16_params_this_group)
        self.fp32_from_float16_groups.append(
            fp32_from_float16_params_this_group)
        self.fp32_from_fp32_groups.append(fp32_params_this_group)

def FP32OptimizerInit(self, optimizer, clip_grad,
             log_num_zeros_in_grad,
             params_have_main_grad,
             use_contiguous_buffers_in_local_ddp,
             models):

    super(FP32Optimizer, self).__init__(
        optimizer, clip_grad, log_num_zeros_in_grad,
        params_have_main_grad, use_contiguous_buffers_in_local_ddp,
        models)

    self._scale = torch.mlu.FloatTensor([1.0])

megatron.optimizer.optimizer.MixedPrecisionOptimizer.__init__ = MixedPrecisionOptimizerInit
megatron.optimizer.optimizer.MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan = _unscale_main_grads_and_check_for_nan
megatron.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__ = Float16OptimizerWithFloat16ParamsInit
megatron.optimizer.optimizer.FP32Optimizer.__init__ = FP32OptimizerInit 
