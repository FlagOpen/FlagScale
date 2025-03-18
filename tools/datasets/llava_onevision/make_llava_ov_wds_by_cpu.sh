#!/bin/bash

LLaVA_NeXT_HOME="Path_Of_LLaVA-NeXT"
VISION_MODEL_PATH="Path_Of_VISION_MODEL"
PROMPT_VERSION="qwen_1_5"

# # stage1
# image_aspect_ratio=square

# other stages
image_aspect_ratio=anyres_max_9

set -u
  DATA_PATH=$1
  EXPNAME_PATH=$2
  HOSTFILE=$3
set +u

echo "BASE_RUN_NAME: ${EXPNAME_PATH}"

CKPT_PATH="./checkpoints"

mkdir -p $CKPT_PATH
mkdir -p $EXPNAME_PATH
LOGFILE=$EXPNAME_PATH/exp.log
NNodes=`wc -l ${HOSTFILE} | cut -d " " -f1`
echo "nnodes: ${NNodes}"

mpirun --hostfile ${HOSTFILE} -np ${NNodes} --allow-run-as-root --map-by node \
python llava_ov_wds_by_cpu.py \
  --model_name_or_path ${CKPT_PATH} \
  --version ${PROMPT_VERSION} \
  --data_path $DATA_PATH \
  --image_folder playground/data \
  --video_folder ./onevision_data/videos \
  --mm_tunable_parts="mm_mlp_adapter" \
  --mm_vision_tower_lr=2e-6 \
  --vision_tower ${VISION_MODEL_PATH} \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --mm_spatial_pool_mode "bilinear" \
  --group_by_modality_length True \
  --image_aspect_ratio ${image_aspect_ratio} \
  --image_grid_pinpoints '(1x1),...,(6x6)' \
  --mm_patch_merge_type spatial_unpad \
  --bf16 True \
  --run_name $EXPNAME_PATH \
  --output_dir "${EXPNAME_PATH}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 20 \
  --learning_rate 1e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --model_max_length 32768 \
  --gradient_checkpointing True \
  --dataloader_num_workers 2 \
  --lazy_preprocess True \
  --torch_compile True \
  --torch_compile_backend "inductor" \
  --dataloader_drop_last True \
  --seed 42 \
  --do_train False \
  --frames_upbound 32 1>$LOGFILE.$ip 2>&1 &
