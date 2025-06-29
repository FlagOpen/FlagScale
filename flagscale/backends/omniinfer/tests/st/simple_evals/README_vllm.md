# Simple_evals
支持在线评测vllm的openai接口服务

## 环境准备
```shell
bash build.sh
```

## 文件目录说明
- dataset：存放评测数据集，请勿修改
- results：存放评测结果，模型输入输出详细结果存放于.html文件中，汇总数据存放于.json文件中

## 启动simple_evals评测
```shell
python simple_evals.py --dataset mmlu gpqa mgsm drop humaneval \
--served-model-name ${served_model_name} \
--url "http://localhost:8192/v1" \
--max-tokens 2048 \
--temperature 0.5 \
--num-threads 50 \
--debug
```
参数说明：
- --dataset：评测数据集
- --served-model-name：对应openai标准接口的“model”请求参数
- --url：请求url
- --max-tokens：输出最大tokens数
- --temperature：温度采样参数
- --num-threads：发送评测请求的并发数
- --debug：调试评测脚本，推荐第一次使用时指定，用于判断当前环境和命令是否能正常执行，正式评测前不指定该参数