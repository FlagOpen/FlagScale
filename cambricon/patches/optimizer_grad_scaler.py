import torch
import torch_mlu
import megatron
from megatron.optimizer.grad_scaler import MegatronGradScaler, DynamicGradScaler

def MegatronGradScalerInit(self, initial_scale):
    """Initialize scale value with the input initial scale."""
    assert initial_scale > 0.0
    self._scale = torch.mlu.FloatTensor([initial_scale])

def DynamicGradScalerInit(self, initial_scale, min_scale,
             growth_factor, backoff_factor,
             growth_interval, hysteresis):
    """"Grad scaler with dynamic scale that gets adjusted
    during training."""
    super(DynamicGradScaler, self).__init__(initial_scale)

    # Lower bound on the scale.
    assert min_scale > 0.0
    assert min_scale <= initial_scale
    self.min_scale = torch.mlu.FloatTensor([min_scale])
    # Growth and backoff factors for the scale.
    assert growth_factor > 1.0
    self.growth_factor = torch.mlu.FloatTensor([growth_factor])
    assert backoff_factor < 1.0
    assert backoff_factor > 0.0
    self.backoff_factor = torch.mlu.FloatTensor([backoff_factor])
    # Interval over which if we don't see any inf/nan,
    # we will scale the grad scale by the growth factor.
    assert growth_interval > 0
    self.growth_interval = growth_interval
    # Number of inf/nans we should see before scaling down
    # the grad scale by the backoff factor.
    assert hysteresis > 0
    self.hysteresis = hysteresis

    # Trackers.
    self._growth_tracker = 0
    self._hysteresis_tracker = self.hysteresis

def load_state_dict(self, state_dict):
    self._scale = state_dict['scale'].mlu(torch.mlu.current_device())
    self._growth_tracker = state_dict['growth_tracker']
    self._hysteresis_tracker = state_dict['hysteresis_tracker']

megatron.optimizer.grad_scaler.MegatronGradScaler.__init__ = MegatronGradScalerInit
megatron.optimizer.grad_scaler.DynamicGradScaler.__init__ = DynamicGradScalerInit
megatron.optimizer.grad_scaler.DynamicGradScaler.load_state_dict = load_state_dict 
