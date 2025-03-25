# Use Dockerfile.cuda to build image.

```bash
docker build --build-arg CUDA_VERSION=12.4.1 --build-arg CUDNN_VERSION=9.5.0 --build-arg PYTHON_VERSION=3.12 --build-arg TORCH_VERSION=2.5.1 --build-arg FS_VERSION=${commit id} -f Dockerfile.cuda -t flagscale:cuda-12.4.1-python-3.12-torch-2.5.1-commit-${commit id} .
```

* `CUDA_VERSION`: Can be manually specified, default value is 12.4.1.
* `PYTHON_VERSION`: Can be manually specified, default value is 3.12.
* `TORCH_VERSION`: Can be manually specified, default value is 2.5.1.
* `CUDNN_VERSION`: Can be manually specified, default value is 9.5.0.
* `FS_VERSION`: Must be explicitly specified, enter the FlagScale commit ID, used to mark the FlagScale and environment version corresponding to the image.

# Build an SSH login free image using existing images and Dockerfile.ssh.
NOTE:
   1. This construction method is not secure and is only for internal development use. Do not leak the built image.
   2. It is recommended to rebuild the image for different tasks each time to avoid environmental interference caused by the same key.

```bash
docker build --build-arg BASE_IMAGE=flagscale:cuda-12.4.1-cudnn-9.5.0-python-3.12-torch-2.5.1-commit-${commit id} --build-arg SSH_PORT=22 -f Dockerfile.ssh -t flagscale:cuda-12.4.1-cudnn-9.5.0-python-3.12-torch-2.5.1-commit-${commit id}-ssh .
```

* `BASE_IMAGE`: Must be explicitly specified, enter the name of the base image used.
* `SSH_PORT`: Can be manually specified, default value is 22.
