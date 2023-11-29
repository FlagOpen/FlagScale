import socket
import torch
from functools import cmp_to_key


class HeteroContext:
    def __init__(self, args):
        assert torch.distributed.is_initialized(), 'torch.distributed is not initialized'
        self._hetero_mode = args.hetero_mode
        # The order of device types is very import for creating the logical rank.
        # Users should make sure the order satisfies their needs. 
        self._hetero_device_types = args.hetero_device_types
        self._hetero_current_device_type = args.hetero_current_device_type
        self._hetero_micro_batch_sizes = args.hetero_micro_batch_sizes
        self._hetero_data_parallel_splits = args.hetero_data_parallel_splits
        self._hetero_pipeline_stages = args.hetero_pipeline_stages
        self._hetero_pipeline_stage_splits = args.hetero_pipeline_stage_splits
        self._rank_infos = {}
        self._physical_rank_to_logical_rank = {}
        self._logical_rank_to_physical_rank = {}
        self._build_rank_mapping()

    def _build_rank_mapping(self):
        # Collect all rank infos.
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        all_rank_infos = [None for _ in range(world_size)]
        cur_rank_info = {'rank': rank,
                         'device_type': self._hetero_current_device_type}
        torch.distributed.all_gather_object(
            all_rank_infos, cur_rank_info)

        physical_ranks = []
        for info in all_rank_infos:
            self._rank_infos[info['rank']] = info
            physical_ranks.append(info['rank'])
        
        # Sort the physical ranks by device type and rank.
        def _compare(rank1, rank2):
            device_type1 = self._rank_infos[rank1]['device_type']
            device_type2 = self._rank_infos[rank2]['device_type']
            if self._hetero_device_types \
                and self._hetero_device_types.index(device_type1) < self._hetero_device_types.index(device_type2):
                return -1
            elif self._hetero_device_types \
                and self._hetero_device_types.index(device_type1) > self._hetero_device_types.index(device_type2):
                return 1
            else:
                return rank1 - rank2
        sorted_physical_ranks = sorted(
            physical_ranks, key=cmp_to_key(_compare))

        # Build the mapping between physical rank and logical rank
        for logical_rank, physical_rank in enumerate(sorted_physical_ranks):
            self._physical_rank_to_logical_rank[physical_rank] = logical_rank
            self._logical_rank_to_physical_rank[logical_rank] = physical_rank
        
    def to_physical_ranks(self, logical_ranks):
        physical_ranks = []
        for logical_rank in logical_ranks:
            physical_ranks.append(
                self._logical_rank_to_physical_rank[logical_rank])
        return physical_ranks

    def to_logical_ranks(self, physical_ranks):
        logical_ranks = []
        for physical_rank in physical_ranks:
            logical_ranks.append(
                self._physical_rank_to_logical_rank[physical_rank])
        return logical_ranks

    def __str__(self):
        return (f"HeteroContext( \n"
                f"  hetero_mode={self._hetero_mode}, \n"
                f"  hetero_device_types={self._hetero_device_types}, \n"
                f"  hetero_current_device_type={self._hetero_current_device_type}, \n"
                f"  hetero_micro_batch_sizes={self._hetero_micro_batch_sizes}, \n"
                f"  hetero_data_parallel_splits={self._hetero_data_parallel_splits}, \n"
                f"  hetero_pipeline_stages={self._hetero_pipeline_stages}, \n"
                f"  hetero_pipeline_stage_splits={self._hetero_pipeline_stage_splits}, \n"
                f"  physical_rank_to_logical_rank={self._physical_rank_to_logical_rank}, \n"
                f"  logical_rank_to_physical_rank={self._logical_rank_to_physical_rank}), \n"
                f"  rank_infos={self._rank_infos}")
        


