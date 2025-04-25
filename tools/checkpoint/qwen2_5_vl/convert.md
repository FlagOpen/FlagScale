### 📎 Reference

Mainly based on official [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/examples/qwen2_5_vl),with necessary modifications for integration into the current training framework.

### 下载模型
```bash
mkdir -p /mnt/qwen2.5-vl-ckpts
cd /mnt/qwen2.5-vl-ckpts
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
cd Qwen2.5-VL-7B-Instruct
git lfs pull
```

### Megatron-Core模型格式转换
运行`hf2mcore_qwen2.5_vl_convertor.sh`脚本，需要传入的参数列表如下
```bash
MODEL_SIZE=$1                 # 模型参数：2B/7B/72B
SOURCE_CKPT_PATH=$2           # 源llm checkpoint路径
TARGET_CKPT_PATH=$3           # 目标checkpoint路径
TP=$4                         # 解码器模型并行度
PP=$5                         # 解码器流水并行度
mg2hf=$6                      # 是否执行mcore2hf转换
PR=$7                         # 精度设置，fp16/bf16/fp32
HF_CKPT_PATH=$8               # HF的CKPT的路径【可选，mg2hf=true时必须提供】
```
例如，使用下述脚本将checkpoint转换到MCore-Dense并检查输出

```bash
cd /workspace/FlagScale/tools/checkpointing/qwen2_5_vl
bash hf2mcore_qwen2.5_vl_convertor.sh \
7B \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-7B-Instruct \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-7B-Instruct-tp2pp2 \
2  \
2  \
false \
bf16
```

当您需要将训练好的checkpoint转换回huggingface格式用于推理时，执行

```bash
cd /workspace/FlagScale/tools/checkpointing/qwen2_5_vl
bash hf2mcore_qwen2.5_vl_convertor.sh \
7B \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-7B-Instruct-tp2pp2 \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-7B-Instruct-tp2pp2-back \
2  \
2  \
true \
bf16 \
/mnt/qwen2.5-vl-ckpts/Qwen2.5-VL-7B-Instruct
```

此外，如果您需要在继续预训练时设置不对称PP切分来达到最佳吞吐，在准备模型权重时，与训练阶段类似，您需要手动调整以下环境变量来确定第一个pipeline stage中的Transformer层数
```bash
export MP_PP0_LAYERS=16
```
