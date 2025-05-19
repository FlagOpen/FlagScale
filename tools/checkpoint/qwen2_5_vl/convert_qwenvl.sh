# bash hf2mcore_qwen2.5_vl_convertor.sh 7B \
# /share/project/lizhiyu/FlagScale/train_qwen2_5_vl_7b_data_check_chatml_str/checkpoints/ \
# /share/project/lizhiyu/data/Qwen2.5-VL-7B-Instruct-data-check-same-order \
# 2 1 true bf16  \
# /share/project/lizhiyu/data/Qwen2.5-VL-7B-Instruct


bash hf2mcore_qwen2.5_vl_convertor.sh 7B \
/share/project/lizhiyu/data/Qwen2.5-VL-7B-Instruct_mc_tp2/ \
/share/project/lizhiyu/data/Qwen2.5-VL-7B-Instruct_mc_tp2_hf \
2 1 true bf16  \
/share/project/lizhiyu/data/Qwen2.5-VL-7B-Instruct