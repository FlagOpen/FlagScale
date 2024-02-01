import sys
import time

import torch

import megatron
from megatron import get_args
from megatron.initialize import _warmup_jit_function


def _compile_dependencies():
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling dataset index builder ...")
        from megatron.data.dataset_utils import compile_helper

        compile_helper()
        print(
            ">>> done with dataset index builder. Compilation time: {:.3f} "
            "seconds".format(time.time() - start_time),
            flush=True,
        )


def set_jit_fusion_options():
    # initial mc2
    args = get_args()
    if args.use_npu_mc2:
        from .ascend_turbo.initialize import initialize_cfg_from_args    
        initialize_cfg_from_args(args)
    
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
    _warmup_jit_function()


megatron.initialize._compile_dependencies = _compile_dependencies


for k, v in sys.modules.items():
    if 'megatron' in k and hasattr(v, 'set_jit_fusion_options'):
        setattr(v, 'set_jit_fusion_options', set_jit_fusion_options)
