### 环境变量设置

在使用计算通信并行时，需要开启FFTS和内存复用，分别对应如下这两个环境变量的设置。

```shell
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE # 开启FFTS
export MULTI_STREAM_MEMORY_REUSE=1 # 内存复用

```
