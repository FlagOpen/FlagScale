
export PYTHONPATH=$PYTHONPATH:/share/project/lizhiyu/FlagScale/third_party/Megatron-LM/
# python build_chatml_mp.py \
#     --dataset-root=/share/project/caomingyu/robobrain_train_data/planning_data \
#     --output-root=/share/project/lizhiyu/data/slice_frame \
#     --json=3rscan_plan_11k.json \
#     --images-key=image \
#     --vision-root=/share/project/caomingyu/robobrain_train_images \
#     --shuffle_tars 

# 11701

# python convert_custom_dataset_to_wds_chatml.py \
#     --dataset-root=/share/project/lizhiyu/data/sample_1000/ \
#     --output-root=/share/project/lizhiyu/data/robovqa_1000_cp/ \
#     --json=robovqa_1000_cp.json \
#     --images-key=image \
#     --vision-root=/share/project/caomingyu/robobrain_train_images
#     # --shuffle-tars 
rm -rf /share/project/lizhiyu/data/vg_1000_cp/
python build_chatml_mp.py \
    --dataset-root=/share/project/lizhiyu/data/sample_1000/ \
    --output-root=/share/project/lizhiyu/data/vg_1000_cp/ \
    --json=VG_1000_cp.json \
    --train-split 1 \
    --val-split 0 \
    --images-key=image \
    --vision-root=/share/project/caomingyu/robobrain_train_images
    # --shuffle-tars 