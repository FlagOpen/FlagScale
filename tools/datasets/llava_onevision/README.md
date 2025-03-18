# Introduction

When training the LLava-one-vision model with FlagScale, the original LLava-one-vision dataset needs to be converted to WebDataset format. This tool primarily reuses the data shuffling functionality in the original [LLava-one-vision](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/llava/train/train.py) training trainer, while saving the data in WebDataset format on each rank.

# Usage

## Preparation

1. Download https://github.com/LLaVA-VL/LLaVA-NeXT into Path_Of_LLaVA-NeXT.

2. Download google/siglip-so400m-patch14-384 into VISION_MODEL_PATH.

3. Write a hostfile with one IP per line, like the example below:
```
1.2.3.4 slots=8
1.2.3.5 slots=8
```
4. Prepare a dataset input compatible with the LLava-one-vision library, like next_ov_stage_july21.yaml.

## Example
Directly processing data, but note that the trainer initialization uses the GPU, while the data preprocessing does not actually use the GPU, leading to wasted GPU resources.
'''
  DATA_PATH=next_ov_stage_july21.yaml
  EXPNAME_PATH=*PathOfOutputWebDatasets*
  HOSTFILE=hostfile
  bash make_llava_ov_wds.sh $DATA_PATH $EXPNAME_PATH $HOSTFILE
'''

We currently recommend a more efficient two-stage processing method. In the first step, the GPU is still used, but only for trainer initialization, and the index data of each card is directly saved. In the second step, CPU multiprocessing is used for index processing without occupying GPU resources.

Stage 1: Saving index data with GPU
'''
  DATA_PATH=next_ov_stage_july21.yaml
  EXPNAME_PATH=*PathOfOutputWebDatasets*
  HOSTFILE=hostfile
  bash make_llava_ov_index.sh $DATA_PATH $EXPNAME_PATH $HOSTFILE
'''

Stage 2: Processing index data with CPU
'''
  DATA_PATH=next_ov_stage_july21.yaml
  EXPNAME_PATH=*PathOfOutputWebDatasets*
  HOSTFILE=hostfile
  bash make_llava_ov_wds_by_CPU.sh $DATA_PATH $EXPNAME_PATH $HOSTFILE
'''
