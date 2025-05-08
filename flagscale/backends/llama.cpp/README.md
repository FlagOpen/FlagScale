# Usage
With llama.cpp as inference backend, the model format should be gguf. Llama.cpp provides converting tools, for example:
```sh
cd FlagScale/third_party/llama.cpp
python convert_hf_to_gguf.py /tmp/Qwen3-0.6B/ --outfile /tmp/Qwen3-0.6B/ggml_model_f16.gguf
``` 

# Efficiency Test

Model: https://huggingface.co/Qwen/Qwen3-0.6B

Abbreviations:
- pp512: test prefilling(prompt processing) stage with 512 tokens input
- tg128: test decoding(token generating) stage with 128 tokens output
- t/s: tokens per second
- ngl: number of gpu layers

## Apple M4 (10 core)

### CPU

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 0.6B F16                 |   1.40 GiB |   751.63 M | Metal,BLAS |       4 |           pp512 |        772.66 ± 9.36 |
| qwen3 0.6B F16                 |   1.40 GiB |   751.63 M | Metal,BLAS |       4 |           tg128 |         54.98 ± 2.62 |

### GPU(Metal)

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 0.6B F16                 |   1.40 GiB |   751.63 M | Metal,BLAS |       4 |           pp512 |       2504.29 ± 8.20 |
| qwen3 0.6B F16                 |   1.40 GiB |   751.63 M | Metal,BLAS |       4 |           tg128 |         66.79 ± 0.41 |

## A800, Intel Xeon Platinum 8358 CPU (128 core)

### CPU

| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 0.6B F16                 |   1.40 GiB |   751.63 M | CPU        |      64 |           pp512 |     1088.26 ± 128.44 |
| qwen3 0.6B F16                 |   1.40 GiB |   751.63 M | CPU        |      64 |           tg128 |         58.65 ± 5.64 |

### GPU(CUDA)

| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3 0.6B F16                 |   1.40 GiB |   751.63 M | CUDA       |  99 |           pp512 |    24335.09 ± 171.34 |
| qwen3 0.6B F16                 |   1.40 GiB |   751.63 M | CUDA       |  99 |           tg128 |        168.67 ± 0.53 |