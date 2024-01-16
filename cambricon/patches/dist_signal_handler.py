import torch
import torch_mlu
import megatron

def get_device(local_rank=None):
    backend = torch.distributed.get_backend()
    if backend == 'cncl':
        if local_rank is None:
            device = torch.device('mlu')
        else:
            device = torch.device(f'mlu:{local_rank}')
    elif backend == 'gloo':
        device = torch.device('cpu')
    else:
        raise RuntimeError
    return device

megatron.dist_signal_handler.get_device = get_device
