# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
from megatron.training import get_args


def group_mlp_forward_detach(self, permuted_local_hidden_states, tokens_per_expert):
    args = get_args()
    # is_recompute_activation = args.moe_zero_memory == 'level0' or should_recompute_activation(self.layer_number)
    is_recompute_activation = False

    assert torch.count_nonzero(tokens_per_expert) == 0
    # Make sure parameters still have gradients when no tokens are routed to this set of experts.
    w1 = self.weight1.view(self.config.hidden_size, -1)
    w2 = self.weight2.view(-1, self.config.hidden_size)
    fc1_output = torch.matmul(permuted_local_hidden_states, w1)
    intermediate_parallel = self.activation_func(fc1_output)
    fc2_output = torch.matmul(intermediate_parallel, w2)
    if is_recompute_activation:
        intermediate_parallel.untyped_storage().resize_(0)


    return (fc2_output, fc1_output, intermediate_parallel), None