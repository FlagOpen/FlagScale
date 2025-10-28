Tips: robotics git repo should be added as a submodule, but it is not published yet.

# Training
##  Install FlagScale 
Install training env according to README.md under project root path.
FlagScale training env include:
1. python deps
2. unpatching
3. Megatron-LM
4. Megatron-Energon


## Install Robotics Deps
1. Robotics, not publish yet
2. Lerobot, https://github.com/huggingface/lerobot
3. openpi-client
4. python deps, FlagScale/requirements/train/robotics/requirements.txt


``` sh
# install lerobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .

# install openpi
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
# Or if you already cloned the repo:
# git submodule update --init --recursive
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd packages/openpi-client
pip install -e .

# install other python deps
pip install -r requirements/train/robotics/requirements.txt

```


## Prepare Dataset
Convert WebDataset to Energon:
```
cd FlagScale/tools/datasets/qwenvl/
python convert_robotics.py \
    --dataset-root="" \
    --output-root=/output/path \
    --json=/webdataset/path \
    --train-split 1 \
    --val-split 0 \
    --images-key=image \
    --videos-key=video \
    --actions-key=action \
    --state-key=state \
    --actions-qpos-key=qpos \
    --actions-eepose-key=eepose \
    --state-qpos-key=qpos \
    --state-eepose-key=eepose \
    --vision-root="" \
    --max-samples-per-tar 1000000 \
    --dp-size 1
```

## Edit Config
Config files:
1. examples/robotics/conf/train.yaml
2. examples/robotics/conf/train/3_3b.yaml

Attributes need to change:
1. PYTHONPATH, including Robotics and Lerobot
2. HF_LEROBOT_HOME
3. ENERGON_DATA_PATH, the ernergon dataset

## Start Training
```
cd FlagScale/
python run.py --config-path ./examples/robotics/conf --config-name train action=start
```

# Serving

##  Install FlagScale 
Install serving env according to [README](https://github.com/FlagOpen/FlagScale/blob/main/README.md) 

## Edit Config
Config files: examples/robotics/conf/serve/3_3b.yaml
Attributes need to change:: model path

## Start Serving
```
cd FlagScale/
python run.py --config-path ./examples/robotics/conf --config-name serve action=start
```