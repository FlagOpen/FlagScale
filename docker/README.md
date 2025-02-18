# Use Dockerfile.cuda to build image sample instructions.

```bash
docker build --build-arg CUDA_VERSION=12.4.1 --build-arg PYTHON_VERSION=3.12 --build-arg TORCH_VERSION=2.5.1 --build-arg FS_VERSION=${commit id} -f Dockerfile.cuda -t flagscale:cuda-12.4.1-python-3.12-torch-2.5.1-commit-${commit id} .
```

* `CUDA_VERSION`: Can be manually specified, default value is 12.4.1.
* `PYTHON_VERSION`: Can be manually specified, default value is 3.12.
* `TORCH_VERSION`: Can be manually specified, default value is 2.5.1.
* `FS_VERSION`: Must be explicitly specified, enter the FlagScale commit ID, used to mark the FlagScale and environment version corresponding to the image.
