# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import queue

### Fake WeightGradStore
### WeightGradStore is not used, we choose to use TransformerEngine's dw detach method
### Refs pr: https://github.com/NVIDIA/TransformerEngine/pull/1653


class WeightGradStore:
    cache = []
    weight_grad_queue = queue.Queue()
    store_grad_cache = []
    grad_store = []
    gather_stream = None
    is_decoupleBlock = False

    @classmethod
    def put(cls, total_input, grad_output, weight, sequence_parallel, in_row=False):
        # cls.cache.append((total_input, grad_output, weight, sequence_parallel, in_row))
        return

    @classmethod
    def flush_chunk_grad(cls):
        # cls.weight_grad_queue.put(cls.cache)
        # cls.cache = []
        return

    @classmethod
    def start_decouple(cls):
        cls.is_decoupleBlock = True

    @classmethod
    def end_decouple(cls):
        cls.is_decoupleBlock = False


    @classmethod
    def pop(cls, overlap_arg=None):
        return 

    @classmethod
    def pop_single(cls):
        return 