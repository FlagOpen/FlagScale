
export PYTHONPATH=$PYTHONPATH:/share/project/lizhiyu/FlagScale/third_party/Megatron-LM/

# rm -rf /share/project/lizhiyu/data/robovqa_1000_cp/
# python convert_custom_dataset_to_wds_chatml_str.py \
#     --dataset-root=/share/project/lizhiyu/data/sample_1000/ \
#     --output-root=/share/project/lizhiyu/data/robovqa_1000_cp/ \
#     --json=robovqa_1000_cp.json \
#     --train-split 1 \
#     --val-split 0 \
#     --images-key=image \
#     --max-samples-per-tar 10000000 \
#     --vision-root=/share/project/caomingyu/robobrain_train_images

# python convert_custom_dataset_to_wds_chatml_str.py \
#     --dataset-root=/share/project/caomingyu/data_check/ \
#     --output-root=/share/project/lizhiyu/data/data_check_200k_str_one_tar/ \
#     --json=data_check_200k.json \
#     --train-split 1 \
#     --val-split 0 \
#     --images-key=image \
#     --vision-root=/share/project/caomingyu/robobrain_train_images \
#     --max-samples-per-tar 10000000 \
#     --dp-size 4

# python convert_custom_dataset_to_wds_chatml_str.py \
#     --dataset-root=/share/project/caomingyu/data_check/ \
#     --output-root=/share/project/lizhiyu/data/data_check_200k_str_one_tar_tp1/ \
#     --json=data_check_200k.json \
#     --train-split 1 \
#     --val-split 0 \
#     --images-key=image \
#     --vision-root=/share/project/caomingyu/robobrain_train_images \
#     --max-samples-per-tar 10000000 \
#     --dp-size 8

# python convert_custom_dataset_to_wds_chatml_str.py \
#     --dataset-root=/share/project/caomingyu/temp/ \
#     --output-root=/share/project/lizhiyu/data/sample_data/absxy_20k/ \
#     --json=region_ref_max_points_10_absxy_20k.json \
#     --train-split 1 \
#     --val-split 0 \
#     --images-key=image \
#     --vision-root=/share/project/caomingyu/temp/region_ref \
#     --max-samples-per-tar 10000000 \
#     --dp-size 8


# python convert_custom_dataset_to_wds_chatml_str.py \
#     --dataset-root=/share/project/lizhiyu/data \
#     --output-root=/share/project/lizhiyu/data/sample_data/final_interleaved_large_order_preserved_small_shuffled/ \
#     --json=final_interleaved_large_order_preserved_small_shuffled.json \
#     --train-split 1 \
#     --val-split 0 \
#     --images-key=image \
#     --vision-root=/share/project/caomingyu/robobrain_train_images \
#     --max-samples-per-tar 4056776 \
#     --dp-size 32

python convert_custom_dataset_to_wds_chatml_str.py \
    --dataset-root=/share/project/lizhiyu/data \
    --output-root=/share/project/lizhiyu/data/sample_data/VG_280K/ \
    --json=VG_280K.json \
    --train-split 1 \
    --val-split 0 \
    --images-key=image \
    --videos-key=video \
    --vision-root=/share/project/caomingyu/robobrain_train_images \
    --max-samples-per-tar 4056776 \
    --dp-size 16