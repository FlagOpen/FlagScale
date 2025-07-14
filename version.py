import platform
import subprocess
import sys

from packaging.version import parse

FLAGSCALE_VERSION = "0.8.0"

BACKEND_DEVICE_VERSION = {
    "vllm": {
        "gpu": [
            "0.9.2.dev0+gb6553be1b.d20250708.cu124-cp310-cp310-linux_x86_64",
            "0.9.2.dev0+gb6553be1b.d20250708.cu124-cp312-cp312-linux_x86_64",
        ]
    }
}


def get_whl_version(backend, device):
    if device == "gpu":
        return _get_whl_version_gpu(backend)
    else:
        raise NotImplementedError(f"Unsupported device type: {device}")


def _get_whl_version_gpu(backend):
    def _get_cuda_version():
        """Get the CUDA version from nvcc.
        Adapted from https://github.com/vllm-project/vllm/blob/main/setup.py
        """
        from torch.utils.cpp_extension import CUDA_HOME

        assert CUDA_HOME is not None, "CUDA_HOME is not set"
        nvcc_output = subprocess.check_output(
            [CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True
        )
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        nvcc_cuda_version = parse(output[release_idx].split(",")[0])
        nvcc_cuda_version = "cu" + str(nvcc_cuda_version).replace(".", "")
        return nvcc_cuda_version

    cuda_version = _get_cuda_version()
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_str = platform.system().lower()  # 'linux' / 'darwin' / 'windows'
    arch = platform.machine().lower()  # 'x86_64' / 'aarch64' / etc.
    current_platform_tag = f"{platform_str}_{arch}"

    if backend == "vllm":
        suffix = f"{cuda_version}-{python_version}-{python_version}-{current_platform_tag}"
        versions = BACKEND_DEVICE_VERSION["vllm"]["gpu"]
        compatible_versions = []
        for version in versions:
            if version.endswith(suffix):
                compatible_versions.append(version)
        return versions, compatible_versions
    else:
        raise NotImplementedError(f"Unsupported backend: {backend}")
