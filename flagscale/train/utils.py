import sys
import importlib
import os

import torch
from megatron.core import mpu

_hook_modules = {"transformer_engine"}

device_type = os.environ.get('DEVICE_TYPE',None)

class Empty:
    def __init__(self,*args):    
       pass

def module_hook(fullname, module):
    if device_type == 'iluvatar':
        if fullname == "transformer_engine":
            print("module_hook fullname:", fullname)
            module.common.recipe = Empty()
            module.common.recipe.DelayedScaling = Empty()

class CustomModuleFinder:

    def find_module(self, fullname, path=None):
        if fullname in _hook_modules:
            return CustomModuleLoader()

class CustomModuleLoader:

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        finder = sys.meta_path.pop(0)
        module = importlib.import_module(fullname)
        module_hook(fullname, module)
        sys.meta_path.insert(0, finder)
        return module

def get_batch_on_this_tp_rank(data_iterator, args=None):
    """Get a batch of data on the current tensor model parallel rank for heterogenous model parallelism."""
    
    def _broadcast(item):
       if item is not None:
           torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

       if data_iterator is not None:
           data = next(data_iterator)
       else:
           data = None

       batch = {
           'tokens': data["tokens"].cuda(non_blocking = True),
           'labels': data["labels"].cuda(non_blocking = True),
           'loss_mask': data["loss_mask"].cuda(non_blocking = True),
           'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking = True),
           'position_ids': data["position_ids"].cuda(non_blocking = True)
       }

       if args.pipeline_model_parallel_size == 1:
           _broadcast(batch['tokens'])
           _broadcast(batch['labels'])
           _broadcast(batch['loss_mask'])
           _broadcast(batch['attention_mask'])
           _broadcast(batch['position_ids'])

       elif mpu.is_pipeline_first_stage():
           _broadcast(batch['tokens'])
           _broadcast(batch['attention_mask'])
           _broadcast(batch['position_ids'])

       elif mpu.is_pipeline_last_stage():
           _broadcast(batch['labels'])
           _broadcast(batch['loss_mask'])
           _broadcast(batch['attention_mask'])

    else:
       cur_mc_size = args.micro_batch_size * args.data_parallel_size // mpu.get_data_parallel_world_size()
       tokens=torch.empty((cur_mc_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
       labels=torch.empty((cur_mc_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
       loss_mask=torch.empty((cur_mc_size,args.seq_length), dtype = torch.float32 , device = torch.cuda.current_device())
       if args.create_attention_mask_in_dataloader:
           attention_mask=torch.empty(
                (cur_mc_size,1,args.seq_length,args.seq_length), dtype = torch.bool , device = torch.cuda.current_device()
            )
       else:
           attention_mask=None
       position_ids=torch.empty((cur_mc_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())

       if args.pipeline_model_parallel_size == 1:
           _broadcast(tokens)
           _broadcast(labels)
           _broadcast(loss_mask)
           _broadcast(attention_mask)
           _broadcast(position_ids)
 
       elif mpu.is_pipeline_first_stage():
           labels=None
           loss_mask=None
   
           _broadcast(tokens)
           _broadcast(attention_mask)
           _broadcast(position_ids)

       elif mpu.is_pipeline_last_stage():
           tokens=None
           position_ids=None
    
           _broadcast(labels)
           _broadcast(loss_mask)
           _broadcast(attention_mask)
 
       batch = {
           'tokens': tokens,
           'labels': labels,
           'loss_mask': loss_mask,
           'attention_mask': attention_mask,
           'position_ids': position_ids
       }

    return batch