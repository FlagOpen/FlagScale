import os
import torch
import torch_mlu

# megatron/core/
from . import core_models_common_rotary_pos_embedding
from . import core_pipeline_parallel_schedules 
from . import core_tensor_parallel_data
from . import core_tensor_parallel_layer
from . import core_tensor_parallel_random

# megatron/optimizer/
from . import optimizer_grad_scaler
from . import optimizer_clip_grads
from . import optimizer_distrib_optimizer 
from . import optimizer_optimizer 

# megatron/model/
from . import model_transformer

# megatron/
from . import dist_signal_handler
from . import checkpointing
from . import initialize
from . import training 

os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
torch.Tensor.cuda = torch.Tensor.mlu
torch.cuda.DoubleTensor = torch.mlu.DoubleTensor
torch.cuda.FloatTensor = torch.mlu.FloatTensor
torch.cuda.LongTensor = torch.mlu.LongTensor
torch.cuda.HalfTensor = torch.mlu.HalfTensor
torch.cuda.BFloat16Tensor = torch.mlu.BFloat16Tensor
torch.cuda.IntTensor = torch.mlu.IntTensor
torch.cuda.current_device = torch.mlu.current_device
torch.cuda.device_count = torch.mlu.device_count
torch.cuda.set_device = torch.mlu.set_device
torch.cuda.synchronize = torch.mlu.synchronize
torch.cuda.get_rng_state = torch.mlu.get_rng_state
torch.cuda.memory_allocated = torch.mlu.memory_allocated
torch.cuda.max_memory_allocated = torch.mlu.max_memory_allocated
torch.cuda.memory_reserved = torch.mlu.memory_reserved
torch.cuda.max_memory_reserved = torch.mlu.max_memory_reserved
torch.cuda.memory_stats = torch.mlu.memory_stats
torch.cuda.empty_cache = torch.mlu.empty_cache
