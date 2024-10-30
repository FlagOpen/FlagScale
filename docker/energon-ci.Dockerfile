FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV TORCH_CUDA_ARCH_LIST "7.0;7.5;8.0;8.6;8.7;8.9;9.0"
ENV MMCV_WITH_OPS 1
ENV FORCE_CUDA 1

RUN python3 -m pip install --upgrade pip

# Install, then uninstall to get only the deps
COPY . ./megatron-energon
RUN pip install -e ./megatron-energon && pip uninstall -y megatron-energon && rm -rf ./megatron-energon
