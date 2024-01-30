# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
import pathlib
import subprocess

from torch.utils import cpp_extension

# Setting this param to a list has a problem of generating different
# compilation commands (with diferent order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicity in
# extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def load(args):

    # Check if cuda 11 is installed for compute capability 8.0
    # From DCU
    # cc_flag = []
    # _, bare_metal_major, bare_metal_minor = _get_cuda_bare_metal_version(
    #     cpp_extension.CUDA_HOME
    # )
    # if int(bare_metal_major) >= 11:
    #     cc_flag.append('-gencode')
    #     cc_flag.append('arch=compute_80,code=sm_80')
    #     if int(bare_metal_minor) >= 8:
    #         cc_flag.append('-gencode')
    #         cc_flag.append('arch=compute_90,code=sm_90')

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / "build"
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    # From DCU
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=[
                "-O3",'-D__HIP_PLATFORM_AMD__',
            ],
            extra_cuda_cflags=[
                "-O3",'-D__HIP_PLATFORM_AMD__'
            ]
            + extra_cuda_flags,
            verbose=(args.rank == 0),
        )

# From DCU
def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/hipcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("version:") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")
