

#  Install FlagScale 
Install inference env according to [README](https://github.com/FlagOpen/FlagScale/blob/main/README.md) 

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
1. Configure video backend as pyav. Configuration example [here]https://github.com/FlagOpen/FlagScale/blob/main/examples/pi0/conf/inference/pi0.yaml).
2. Check [this](https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec) for the compatibility between versions of torchcodec, torch and Python.

# Run Inference Example
```sh
cd FlagScale/
python run.py --config-path ./examples/pi0/conf --config-name inference action=stop
```

