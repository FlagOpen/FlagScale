import torch
import torch_musa
import megatron
from megatron.core.dist_checkpoint.mapping import ShardedTensor, is_main_replica

def _save_to_existing_array(sharded_tensor: ShardedTensor, arr: zarr.Array):
    if not is_main_replica(sharded_tensor.replica_id):
        return
    x = sharded_tensor.data
    x = x.detach().cpu()
    torch.musa.synchronize()
    if x.dtype == torch.bfloat16:
        x = x.float()
        x = x.numpy()
        x = x.astype('bfloat16')
    else:
        x = x.numpy()

    if sharded_tensor.flattened_range is None:
        arr[sharded_tensor.global_slice()] = x
    else:
        arr.set_coordinate_selection(sharded_tensor.global_coordinates(), x)

megatron.core.dist_checkpointing.strategies.zarr._save_to_existing_array = _save_to_existing_array