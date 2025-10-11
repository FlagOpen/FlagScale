

#  Install FlagScale 
If you don't have access to the international internet, clone Flagscale from https://gitee.com/hchnr/flag-scale with branch **lerobot_custom**.

```sh
git clone https://gitee.com/hchnr/flag-scale.git
cd flag-scale/
git checkout lerobot_custom
```

Install inference env according to [README](https://github.com/FlagOpen/FlagScale/blob/main/README.md) 

# Install Lerobot
If you don't have access to the international internet, change submodule third_party/lerobot:

```ini
[submodule "third_party/lerobot"]
	path = third_party/lerobot
	url = https://gitee.com/hchnr/lerobot.git
```

Install lerobot
```sh
cd FlagScale/
git submodule update --init third_party/lerobot
cd third_party/lerobot/
pip install -e .
```

# Install FFmpeg
```sh
conda install ffmpeg -c conda-forge
```

FFmpeg depends torchcodec, which is not easy to install correctly. You can:
1. Configure video backend as pyav. Configuration example [here](https://github.com/FlagOpen/FlagScale/blob/main/examples/pi0/conf/inference/pi0.yaml).
2. Check [this](https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec) for the compatibility between versions of torchcodec, torch and Python.


# Dowmload Model
```sh
modelscope download --model lerobot/pi0 --local_dir /share/pi0
modelscope download --model google/paligemma-3b-pt-224 --local_dir /share/paligemma-3b-pt-224
```

# Edit Config
```sh
cd FlagScale/
vim examples/pi0/conf/inference/pi0.yaml
```
Change 2 fields:
- llm.model_path -> /share/pi0
- llm.tokenizer_path -> /share/paligemma-3b-pt-224


# Download Dataset
If you don't have access to the international internet, prepare dataset first:
```sh
modelscope download --dataset lerobot/aloha_mobile_cabinet --local_dir /share/aloha_mobile_cabinet
mkdir -p ~/.cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet
cp -r /share/aloha_mobile_cabinet/* ~/.cache/huggingface/lerobot/lerobot/aloha_mobile_cabinet
```

# Run Inference Example
```sh
cd FlagScale/
python run.py --config-path ./examples/pi0/conf --config-name inference action=run
```

