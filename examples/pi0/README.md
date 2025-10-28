

#  Install FlagScale

Clone FlagScale code from github.

If you don't have access to the international internet, import FlagScale project on [gitee](https://gitee.com/), then clone from gitee.

```sh
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale/
```

Install train and inference env according to [README](https://github.com/FlagOpen/FlagScale/blob/main/README.md) 

# Download Model

```sh
git lfs install

mkdir -p /share/pi0
cd /share/pi0
git clone https://huggingface.co/lerobot/pi0_base

mkdir -p /share/paligemma-3b-pt-224
cd /share/paligemma-3b-pt-224
git clone https://huggingface.co/google/paligemma-3b-pt-224
```

If you don't have access to the international internet, download from modelscope.

```sh
modelscope download --model lerobot/pi0 --local_dir /share/pi0
modelscope download --model google/paligemma-3b-pt-224 --local_dir /share/paligemma-3b-pt-224
```


# Download Statistics

```sh
mkdir -p /share/lerobot/aloha_mobile_cabinet
cd /share/lerobot/aloha_mobile_cabinet
git clone https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet
```

If you don't have access to the international internet, download from modelscope.

```sh
modelscope download --dataset lerobot/aloha_mobile_cabinet --local_dir /share/lerobot/aloha_mobile_cabinet
```

# Training

## Prepare Dataset

FlagScale uses WebDataset format and Megatraon.Energon data loader, you need process your data first.

For example: [demo_0913_n2](https://gitee.com/hchnr/flag-scale/tree/robotics_dataset/demo_0913_n2/wds-1)

## Edit Config

```sh
cd FlagScale/
vim examples/pi0/conf/train/pi0.yaml
```
Change 4 fields:
- model.checkpoint_dir -> /share/pi0
- model.stat_path -> /share/lerobot/aloha_mobile_cabinet/meta/stats.json
- data.tokenizer_path -> /share/paligemma-3b-pt-224
- data.data_path -> /share/demo_0913_n2/wds-1

## Start Training
```sh
cd FlagScale/
python run.py --config-path ./examples/pi0/conf --config-name train action=run
```

# Inference

## Edit Config

```sh
cd FlagScale/
vim examples/pi0/conf/inference/pi0.yaml
```

Change 3 fields:
- engine.model -> /share/pi0
- engine.stat_path -> /share/lerobot/aloha_mobile_cabinet/meta/stats.json
- engine.tokenizer -> /share/paligemma-3b-pt-224

## Start Inference
```sh
cd FlagScale/
python run.py --config-path ./examples/pi0/conf --config-name inference action=run
```

# Serving

## Edit Config

```sh
cd FlagScale/
vim examples/pi0/conf/serve/pi0.yaml
```

Change 3 fields:
- engine.model -> /share/pi0
- engine.stat_path -> /share/lerobot/aloha_mobile_cabinet/meta/stats.json
- engine.tokenizer -> /share/paligemma-3b-pt-224

## Run Serving

```sh
cd FlagScale/
python run.py --config-path ./examples/pi0/conf --config-name serve action=run
```

# Test Server with Client

Download test images:

```sh
cd FlagScale/
wget https://gitee.com/hchnr/flag-scale/blob/robotics_dataset/orbbec_0_latest.jpg
wget https://gitee.com/hchnr/flag-scale/blob/robotics_dataset/orbbec_1_latest.jpg
wget https://gitee.com/hchnr/flag-scale/blob/robotics_dataset/orbbec_2_latest.jpg
```

Run client:

```sh
cd FlagScale/
python examples/pi0/client_pi0.py \
--host 127.0.0.1 \
--port 5000 \
--base-img orbbec_0_latest.jpg \
--left-wrist-img orbbec_1_latest.jpg \
--right-wrist-img orbbec_2_latest.jpg \
--num-steps 20
```
