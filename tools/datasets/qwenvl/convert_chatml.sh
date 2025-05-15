
export PYTHONPATH=$PYTHONPATH:/share/project/lizhiyu/FlagScale/third_party/Megatron-LM/
python build_chatml_mp.py \
    --dataset-root=/share/project/caomingyu/robobrain_train_data/planning_data \
    --output-root=/share/project/lizhiyu/data/slice_frame \
    --json=3rscan_plan_11k.json \
    --images-key=image \
    --vision-root=/share/project/caomingyu/robobrain_train_images \
    --shuffle_tars 

# 11701