

#  Install FlagScale 
Clone Flagscale from https://gitee.com/hchnr/flag-scale with branch **lerobot_domestic**.

Install inference env according to [README](https://gitee.com/hchnr/flag-scale/blob/main/README.md) 

# Install Lerobot

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
1. Configure video backend as pyav. Configuration example [here](https://gitee.com/hchnr/flag-scale/blob/main/examples/pi0/conf/inference/pi0.yaml).
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

# Run Inference Example
```sh
cd FlagScale/
python run.py --config-path ./examples/pi0/conf --config-name inference action=run
```

