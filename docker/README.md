# Use Dockerfile.cuda to build image.

```bash
docker build --build-arg CUDA_VERSION=12.4.1 --build-arg CUDNN_VERSION=9.8.0 --build-arg PYTHON_VERSION=3.12 --build-arg TORCH_VERSION=2.6.0 -f Dockerfile.cuda -t flagscale:cuda12.4.1-cudnn9.8.0-python3.12-torch2.6.0-time2503251131 .
```

* `CUDA_VERSION` `CUDNN_VERSION` `PYTHON_VERSION` `TORCH_VERSION`: Can be manually specified, default value is set in `docker/Dockerfile.cuda`.
* `time`: Manually input the time of building the image, starting from year and accurate to minute, for example `25(year)03(month)25(day)11(hour)31(minute)`.

# Build an SSH login free image using existing images and Dockerfile.ssh.
NOTE:
   1. This construction method is not secure and is only for internal development use. Do not leak the built image.
   2. It is recommended to rebuild the image for different tasks each time to avoid environmental interference caused by the same key.

```bash
docker build --build-arg BASE_IMAGE=flagscale:cuda12.4.1-cudnn9.8.0-python3.12-torch2.6.0-time2503251131 --build-arg SSH_PORT=22 -f Dockerfile.ssh -t flagscale:cuda12.4.1-cudnn9.8.0-python3.12-torch2.6.0-time2503251131-ssh .
```

* `BASE_IMAGE`: Must be explicitly specified, enter the name of the base image used.
* `SSH_PORT`: Can be manually specified, default value is 22.
