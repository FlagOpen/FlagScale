# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import os

def patch_vllm_distributed():
    from vllm import distributed
    from omni.adaptors.vllm.distributed.parallel_state import (
        initialize_model_parallel,
        GroupCoordinator
    )

    distributed.parallel_state.GroupCoordinator = GroupCoordinator
    distributed.initialize_model_parallel = initialize_model_parallel
    distributed.parallel_state.initialize_model_parallel = initialize_model_parallel
    print("++++++++++++++++++++++++patch_vllm_distributed++++++++++++++++++++++++++")
 
def patch_rope():
    from vllm.model_executor.layers import rotary_embedding
 
    from omni.models.common.layers.rotary_embedding import get_rope
    rotary_embedding.get_rope = get_rope
    print("+++++++++++++++++++++++patch_rope+++++++++++++++++++++++++++")
 
def patch_embedding():
    from vllm.model_executor.layers import vocab_parallel_embedding
    from omni.models.common.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
    vocab_parallel_embedding.VocabParallelEmbedding = VocabParallelEmbedding
    vocab_parallel_embedding.ParallelLMHead = ParallelLMHead
    vocab_parallel_embedding.VocabParallelEmbedding.forward = VocabParallelEmbedding.forward_vocab

def patch_sampler():
    from omni.models.common.layers.sampler import AscendSampler
    from vllm.model_executor.layers import sampler
    sampler.Sampler = AscendSampler
    from vllm.model_executor.layers import rejection_sampler
    from omni.models.common.layers.sampler import RejectionSampler, _multinomial
    rejection_sampler.RejectionSampler = RejectionSampler
    rejection_sampler._multinomial = _multinomial
    print("++++++++++++++++++++++patch_sampler++++++++++++++++++++++++++++")
    

_patch_done = False

def patch_all():
    global _patch_done
    if _patch_done:
        return
    omni_use_dsv3 = int(os.getenv("OMNI_USE_DSV3", "0"))
    patch_vllm_distributed()
    if omni_use_dsv3:
        patch_rope() # this patch need to move to dsv3 path to avoid errors
    patch_embedding()
    patch_sampler()
    _patch_done = True

patch_all() 
