import socket
import torch
from functools import cmp_to_key
from collections import namedtuple
from itertools import product as cartesian_product


class ProcessMesh:
    """ Define n-dimensional Cartesian process topology. """

    def __init__(self, dims, dim_names):
        """Create a mapping of n-dimensional tensor coordinates to linear indices.

        Arguments:
            dims (list): the dimension (length) of each axis of the topology tensor
            dim_names (list): the names of the tensor dim_names
        """

        self._dims = dims 
        self._dim_names = dim_names
        assert all([d > 0 for d in self._dims]), \
            f'dims={self._dims} must be positive integers'
        assert len(self._dims_names) == len(set(self._dim_names)), \
            f'dim_names={self._dim_names} contains duplicates'
        assert len(self._dims) == len(self._dim_names), \
            f'len(dims)={len(self._dims)} != len(dim_names)={len(self._dim_names)}'

        self._rank_to_coord = {}
        self._coord_to_rank = {}

        ranges = [range(d) for d in self._dims]
        for global_rank, coord in enumerate(cartesian_product(*ranges)):
            self._rank_to_coord[global_rank] = tuple(coord)
            self._coord_to_rank[tuple(coord)] = global_rank


    def get_rank(self, coord):
        """Return the global rank of a process via its coordinates."""

        key = tuple(coord)
        assert key in self._coord_to_rank, f'coord {coord} not found in topology.'
        return self._coord_to_rank[key]


    def get_coord(self, rank):
        """Return the coordinate of a global process rank."""

        assert rank in self._rank_to_coord, f'rank {rank} not found in topology.'
        return self._rank_to_coord[rank]

    def __str__(self):
        return "rank_to_coord: " + str(self._rank_to_coord) + "\n" \
            + "coord_to_rank: " + str(self._coord_to_rank)


class HeteroContext:
    def __init__(self, args):
        assert torch.distributed.is_initialized(), 'torch.distributed is not initialized'
        self._world_size = torch.distributed.get_world_size()
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
        self._default_process_mesh = None
        self._new_process_mesh = None
        self._default_rank_to_new_rank = {}
        self._new_rank_to_default_rank = {}
        self._physical_rank_to_logical_rank = {}
        self._logical_rank_to_physical_rank = {}
        # Build the process mesh.
        self._build_process_mesh(args)
        # Build the rank mapping.
        self._build_rank_mapping()
    
    def _build_process_mesh(self, args):
        tensor_model_parallel_size = args.tensor_model_parallel_size 
        pipeline_model_parallel_size = args.pipeline_model_parallel_size 
        context_parallel_size = args.context_parallel_size 
        data_parallel_size = self._world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
        )
        expert_parallel_size = args.expert_parallel_size

        if self._hetero_mode == "pp":
            assert context_parallel_size == 1, \
                f"Context parallel size must be 1 in hetero mode {self._hetero_mode}"
            assert expert_parallel_size == 1, \
                f"Expert parallel size must be 1 in hetero mode {self._hetero_mode}"
            dims = [pipeline_model_parallel_size, data_parallel_size,
                    context_parallel_size, tensor_model_parallel_size]
            dim_names = ['pp', 'dp', 'cp', 'tp']
            self._new_process_mesh = ProcessMesh(dims, dim_names) 
        elif self._hetero_mode == "dp":
            assert context_parallel_size == 1, \
                f"Context parallel size must be 1 in hetero mode {self._hetero_mode}"
            assert expert_parallel_size == 1, \
                f"Expert parallel size must be 1 in hetero mode {self._hetero_mode}"
            dims = [data_parallel_size, context_parallel_size,
                    pipeline_model_parallel_size, tensor_model_parallel_size]
            dim_names = ['dp', 'cp', 'pp', 'tp']
            self._new_process_mesh = ProcessMesh(dims, dim_names) 
        else:
            self._new_process_mesh = self._default_process_mesh 

    def _build_rank_mapping(self):
        # Build the mapping between source rank and target rank.
        for default_rank in range(self._world_size):
            default_coord = self._default_process_mesh.get_coord(default_rank)
            if self._hetero_mode == "pp":
                # pp, dp, cp, tp => pp, dp, cp, tp
                new_coord = (default_coord[0], default_coord[1],
                             default_coord[2], default_coord[3])
            elif self._hetero_mode == "dp":
                # pp, dp, cp, tp => dp, cp, pp, tp
                new_coord = (default_coord[1], default_coord[2],
                             default_coord[0], default_coord[3])
            else:
                # pp, dp, cp, tp => pp, dp, cp, tp 
                new_coord = default_coord
            new_rank = self._new_process_mesh.get_rank(new_coord)
            self._default_rank_to_new_rank[default_rank] = new_rank
            self._new_rank_to_default_rank[new_rank] = default_rank

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
        for default_logical_rank, physical_rank in enumerate(sorted_physical_ranks):
            new_logical_rank = self._default_rank_to_new_rank[default_logical_rank]
            self._physical_rank_to_logical_rank[physical_rank] = new_logical_rank 
            self._logical_rank_to_physical_rank[new_logical_rank] = physical_rank
        
    def to_physical_ranks(self, logical_ranks):
        if self._hetero_mode is None:
            return logical_ranks
        physical_ranks = []
        for logical_rank in logical_ranks:
            physical_ranks.append(
                self._logical_rank_to_physical_rank[logical_rank])
        return physical_ranks

    def to_logical_ranks(self, physical_ranks):
        if self._hetero_mode is None:
            return physical_rank 
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
                f"  default_process_mesh={self._default_process_mesh}, \n"
                f"  new_process_mesh={self._new_process_mesh}, \n"
                f"  default_rank_to_new_rank={self._default_rank_to_new_rank}, \n"
                f"  new_rank_to_default_rank={self._new_rank_to_default_rank}, \n"
                f"  physical_rank_to_logical_rank={self._physical_rank_to_logical_rank}, \n"
                f"  logical_rank_to_physical_rank={self._logical_rank_to_physical_rank}), \n"
                f"  rank_infos={self._rank_infos}")
