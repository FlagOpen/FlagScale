
export PYTHONPATH=$PYTHONPATH:../../../../third_party/Megatron-LM/

python convert_custom_dataset_to_wds_chatml_str.py \
    --dataset-root=/share/project/caomingyu/stage1_train_datav4/ \
    --output-root=/share/project/lizhiyu/data/sample_data/final_train_data/ \
    --json=final_train_data.json \
    --train-split 1 \
    --val-split 0 \
    --images-key=image \
    --videos-key=video \
    --vision-root=/share/project/caomingyu/robobrain_train_images \
    --max-samples-per-tar 100000000 \
    --dp-size 32
