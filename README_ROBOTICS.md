# Quick Start

## Pull Docker Image
```sh
docker pull ghcr.io/robobrain-roboos-robotic/robotics_pretrain_flagscale:cuda12.4.1-cudnn9.5.0-python3.12-torch2.6.0-time250928-ssh
```

## Run Container
```sh
docker run -itd --name robotics_pretrain --privileged --gpus all --net=host --ipc=host --device=/dev/infiniband --shm-size 512g --ulimit memlock=-1 -v /nfs/hcr/models/:/models ghcr.io/robobrain-roboos-robotic/robotics_pretrain_flagscale:cuda12.4.1-cudnn9.5.0-python3.12-torch2.6.0-time250928-ssh
```

## Train
```sh
cd /root/robotics_pretrain/flag-scale
conda activate flagscale-train
python run.py --config-path ./examples/qwen2_5_vl/conf --config-name train_3b_action_S6_subtask_agilex_eval5_demo action=run
```

## Serve & Inference
```sh
python run.py --config-path ./examples/qwen2_5_vl/conf/ --config-name serve_3b action=run
```