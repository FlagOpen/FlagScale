# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.utils import make_viewless_tensor
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

from megatron.core.transformer.moe.moe_utils import (
    get_capacity,
    maybe_move_tensor_to_cpu,
    permute,
    sort_chunks_by_idxs,
    unpermute,
)


def _maybe_dtoh_and_synchronize(
        self, point: str, tokens_per_expert: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Move all possible GPU tensors to CPU and make a synchronization at the expected point.
        """
        if not self.drop_and_pad:
            if point == self.cuda_dtoh_point:
                # Move all possible GPU tensors to CPU at self.cuda_dtoh_point.
                on_side_stream = torch.cuda.current_stream() != self.cuda_dtoh_stream
                if on_side_stream:
                    self.cuda_dtoh_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.cuda_dtoh_stream):
                    # TODO: use MemcpyBatchAsync instead.
                    tokens_per_expert = maybe_move_tensor_to_cpu(
                        tokens_per_expert, record_stream=on_side_stream
                    )
                    self.input_splits = maybe_move_tensor_to_cpu(
                        self.input_splits, as_numpy=True, record_stream=on_side_stream
                    )
                    self.output_splits = maybe_move_tensor_to_cpu(
                        self.output_splits, as_numpy=True, record_stream=on_side_stream
                    )
                    self.output_splits_tp = maybe_move_tensor_to_cpu(
                        self.output_splits_tp, as_numpy=True, record_stream=on_side_stream
                    )
                    self.num_out_tokens = maybe_move_tensor_to_cpu(
                        self.num_out_tokens, record_stream=on_side_stream
                    )
                    if self.num_local_experts > 1 and not self.config.moe_permute_fusion:
                        self.num_global_tokens_per_local_expert = maybe_move_tensor_to_cpu(
                            self.num_global_tokens_per_local_expert, record_stream=on_side_stream
                        )

            if point == self.cuda_sync_point:
                # Synchronize with the dtoh stream at self.cuda_sync_point.
                self.cuda_dtoh_stream.synchronize()

        return tokens_per_expert

def preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
    
    if self.drop_and_pad:
        # Drop and pad the input to capacity.
        num_tokens = routing_map.size(0) * self.config.moe_router_topk
        self.capacity = get_capacity(
            num_tokens=num_tokens,
            num_experts=self.num_experts,
            capacity_factor=self.moe_expert_capacity_factor,
        )
        self.num_out_tokens = self.capacity * self.num_experts
        # [num_local_experts], number of tokens processed by each expert.
        num_tokens_per_local_expert = torch.full(
            (self.num_local_experts,),
            self.capacity * self.tp_size * self.ep_size,
            dtype=torch.long,
        )
        # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
        # to each local expert by all ranks.
        self.num_global_tokens_per_local_expert = torch.full(
            (self.num_experts * self.tp_size,),
            self.capacity,
            dtype=torch.long,
            device=self.permute_idx_device,
        )
        return num_tokens_per_local_expert

    # [num_experts], number of tokens assigned to each expert from the current rank's input.
    num_local_tokens_per_expert = routing_map.sum(dim=0).long()

    if self.config.moe_expert_capacity_factor is not None:
        # Drop tokens to capacity, no padding.
        self.num_out_tokens = num_local_tokens_per_expert.sum()

        # A synchronization is needed before the first permutation
        # to get the `num_out_tokens` CPU value.
        self._maybe_update_cuda_sync_point("before_permutation_1")
    else:
        # Dropless
        self.num_out_tokens = routing_map.size(0) * self.config.moe_router_topk

    if self.ep_size > 1 or self.tp_size > 1:
        # ===================================================
        # Calculate input_splits, output_splits for alltoall/allgather in variable size.
        # ===================================================
        # [ep_size]. Represents the number of tokens sent by the current rank to other
        # EP ranks.
        self.input_splits = num_local_tokens_per_expert.reshape(
            self.ep_size, self.num_local_experts
        ).sum(axis=1)
        # Gather the global distribution of tokens across ranks.
        # num_global_tokens_per_expert represents the number of tokens sent to each
        # expert by all ranks.
        # [tp_size, ep_size, num_experts]
        num_global_tokens_per_expert = (
            gather_from_sequence_parallel_region(
                num_local_tokens_per_expert, group=self.tp_ep_group
            )
            .reshape(self.ep_size, self.tp_size, self.num_experts)
            .transpose(0, 1)
        )
        # [tp_size, ep_size, num_experts] -> [tp_size, ep_size, num_local_experts]
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()
        # [tp_size, ep_size, num_local_experts] -> [tp_size, ep_size]
        num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(axis=2)
        # [tp_size, ep_size] -> [ep_size]
        # self.output_splits represents the number of tokens received by the current rank
        # from other EP rank.
        self.output_splits = num_global_tokens_per_rank[self.tp_rank]
        # [tp_size, ep_size] -> [tp_size]
        # self.output_splits_tp represents the number of tokens received by the current
        # rank from other TP rank.
        self.output_splits_tp = num_global_tokens_per_rank.sum(axis=1)
        # [tp_size, ep_size, num_local_experts] -> [num_local_experts]
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=(0, 1))

        # A synchronization is needed before expert parallel AlltoAll communication
        # to get the `input_splits` and `output_splits` CPU values.
        self._maybe_update_cuda_sync_point("before_ep_alltoall")
    else:
        num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
            self.num_experts
        )
        num_tokens_per_local_expert = num_local_tokens_per_expert

        # A synchronization is needed before the returns
        # to get the `num_tokens_per_local_expert` CPU value.
        self._maybe_update_cuda_sync_point("before_finish")

    if self.num_local_experts > 1:
        # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
        # to each local expert by all ranks.
        self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
            -1, self.num_local_experts
        )
        if not self.config.moe_permute_fusion:
            # A synchronization is needed before permutation 2
            # to get the `num_global_tokens_per_local_expert` CPU value.
            self._maybe_update_cuda_sync_point("before_permutation_2")

    assert (
        self.cuda_sync_point_priority[self.cuda_dtoh_point]
        <= self.cuda_sync_point_priority[self.cuda_sync_point]
    ), "cuda_sync_point must be after cuda_dtoh_point."
    return num_tokens_per_local_expert


def alltoall_token_perm1(
    self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor,
):
    # Preprocess: Get the metadata for communication, permutation and computation operations.
    self.hidden_shape = hidden_states.shape
    self.probs = probs
    self.routing_map = routing_map
    assert probs.dim() == 2, "Expected 2D tensor for probs"
    assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
    assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
    tokens_per_expert = preprocess(self, routing_map)

    # Permutation 1: input to AlltoAll input
    tokens_per_expert = _maybe_dtoh_and_synchronize(
        self, "before_permutation_1", tokens_per_expert
    )
    self.hidden_shape_before_permute = hidden_states.shape
    (
        permutated_local_input_tokens,
        permuted_probs,
        self.reversed_local_input_permutation_mapping,
    ) = permute(
        hidden_states,
        routing_map,
        probs=probs,
        num_out_tokens=self.num_out_tokens,
        fused=self.config.moe_permute_fusion,
        drop_and_pad=self.drop_and_pad,
    )

    # Perform expert parallel AlltoAll communication
    tokens_per_expert = _maybe_dtoh_and_synchronize(
        self, "before_ep_alltoall", tokens_per_expert
    )

    return permutated_local_input_tokens, permuted_probs, tokens_per_expert


def alltoall_token_perm2(self, global_input_tokens, global_probs, tokens_per_expert):

    if self.tp_size > 1:
        if self.output_splits_tp is None:
            output_split_sizes = None
        else:
            output_split_sizes = self.output_splits_tp.tolist()
        global_input_tokens = gather_from_sequence_parallel_region(
            global_input_tokens, group=self.tp_group, output_split_sizes=output_split_sizes
        )
        global_probs = gather_from_sequence_parallel_region(
            global_probs, group=self.tp_group, output_split_sizes=output_split_sizes
        )

    # Permutation 2: Sort tokens by local expert.
    tokens_per_expert = self._maybe_dtoh_and_synchronize(
        "before_permutation_2", tokens_per_expert
    )
    if self.num_local_experts > 1:
        if self.drop_and_pad:
            global_input_tokens = (
                global_input_tokens.view(
                    self.tp_size * self.ep_size,
                    self.num_local_experts,
                    self.capacity,
                    *global_input_tokens.size()[1:],
                )
                .transpose(0, 1)
                .contiguous()
                .flatten(start_dim=0, end_dim=2)
            )
            global_probs = (
                global_probs.view(
                    self.tp_size * self.ep_size,
                    self.num_local_experts,
                    self.capacity,
                    *global_probs.size()[1:],
                )
                .transpose(0, 1)
                .contiguous()
                .flatten(start_dim=0, end_dim=2)
            )
        else:
            global_input_tokens, global_probs = sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
                probs=global_probs,
                fused=self.config.moe_permute_fusion,
            )

    tokens_per_expert = self._maybe_dtoh_and_synchronize("before_finish", tokens_per_expert)

    return global_input_tokens, tokens_per_expert, global_probs


def alltoall_token_unperm1(
    self,
    hidden_states: torch.Tensor,
    bias: torch.Tensor = None,
):
    assert bias is None, "Bias is not supported in MoEAlltoAllTokenDispatcher"

    # Unpermutation 2: Unsort tokens by local expert.
    if self.num_local_experts > 1:
        if self.drop_and_pad:
            hidden_states = (
                hidden_states.view(
                    self.num_local_experts,
                    self.tp_size * self.ep_size,
                    self.capacity,
                    *hidden_states.size()[1:],
                )
                .transpose(0, 1)
                .contiguous()
                .flatten(start_dim=0, end_dim=2)
            )
        else:
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert.T.ravel(),
                self.restore_output_by_local_experts,
                fused=self.config.moe_permute_fusion,
            )

    if self.tp_size > 1:
        if self.output_splits_tp is None:
            input_split_sizes = None
        else:
            input_split_sizes = self.output_splits_tp.tolist()
        # The precision of TP reduce_scatter should be the same as the router_dtype
        hidden_states = reduce_scatter_to_sequence_parallel_region(
            hidden_states.to(self.probs.dtype),
            group=self.tp_group,
            input_split_sizes=input_split_sizes,
        ).to(hidden_states.dtype)


    return hidden_states


def alltoall_token_unperm2(self, permutated_local_input_tokens):
    # Unpermutation 1: AlltoAll output to output

   # Unpermutation 1: AlltoAll output to output
    output = unpermute(
        permutated_local_input_tokens,
        self.reversed_local_input_permutation_mapping,
        restore_shape=self.hidden_shape_before_permute,
        routing_map=self.routing_map,
        fused=self.config.moe_permute_fusion,
        drop_and_pad=self.drop_and_pad,
    )

    # Reshape the output tensor
    output = output.view(self.hidden_shape)

    return output, None
