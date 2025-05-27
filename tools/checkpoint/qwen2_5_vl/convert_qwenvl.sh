# bash hf2mcore_qwen2.5_vl_convertor.sh 7B \
# /share/project/lizhiyu/data/ckpt/Qwen2.5-VL-7B-Instruct \
# /share/project/lizhiyu/data/ckpt/Qwen2.5-VL-7B-Instruct-tp2 \
# 2 1 false bf16  \
# /share/project/lizhiyu/data/ckpt/Qwen2.5-VL-7B-Instruct


# bash hf2mcore_qwen2.5_vl_convertor.sh 7B \
# /share/project/lizhiyu/data/Qwen2.5-VL-7B-Instruct_mc_tp2/ \
# /share/project/lizhiyu/data/Qwen2.5-VL-7B-Instruct_mc_tp2_hf \
# 2 1 true bf16  \
# /share/project/lizhiyu/data/Qwen2.5-VL-7B-Instruct


bash hf2mcore_qwen2.5_vl_convertor.sh 32B \
/share/project/lizhiyu/data/ckpt/Qwen2.5-VL-32B-Instruct \
/share/project/lizhiyu/data/ckpt/Qwen2.5-VL-32B-Instruct-tp8 \
8 1 false bf16  \
/share/project/lizhiyu/data/ckpt/Qwen2.5-VL-32B-Instruct