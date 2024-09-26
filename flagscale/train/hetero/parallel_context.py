import os
import math
import warnings
import itertools
import operator
from typing import List, Optional
from datetime import timedelta
from functools import cmp_to_key
from collections import defaultdict

import torch

def get_nccl_options(pg_name, nccl_comm_cfgs):
    from megatron.core.parallel_state import get_nccl_options 
    return get_nccl_options(pg_name, nccl_comm_cfgs)


def find_overlapped_mapping(dim1, dim2, global_size=None):
    """
    Finds the overlapped mapping between two dimensions within an optional global size. Please refer to https://eli.thegreenplace.net/2008/08/15/intersection-of-1d-segments for details.
    """
    # Calculate the least common multiple (LCM) of dim1 and dim2, or use global_size if provided
    dim_lcm = global_size if global_size else math.lcm(dim1, dim2)
    # Generate segments for dim1 and dim2
    dim1_segments_len = dim_lcm // dim1
    dim1_segments = [(i * dim1_segments_len, (i + 1) * dim1_segments_len) for i in range(dim1)]
    dim2_segments_len = dim_lcm // dim2
    dim2_segments = [(i * dim2_segments_len, (i + 1) * dim2_segments_len) for i in range(dim2)]
    # Initialize the mapping of overlapped segments
    overlapped_mapping = {i: [] for i in range(dim1)}
    # Calculate overlaps between dim1 and dim2 segments
    for i, (start1, end1) in enumerate(dim1_segments):
        for j, (start2, end2) in enumerate(dim2_segments):
            if start1 < end2 and end1 > start2:  # Check if segments overlap
                # Calculate the overlap offsets relative to the start of the dim1 segment
                local_overlap_start1 = max(start1, start2) - start1
                local_overlap_end1 = min(end1, end2) - start1
                overlapped_mapping[i].append(
                    (j, local_overlap_start1, local_overlap_end1)
                )
    return overlapped_mapping


class RankMapper:
    def __init__(self, args):
        assert (
            torch.distributed.is_initialized()
        ), "torch.distributed is not initialized"
        self._world_size = torch.distributed.get_world_size()
        # The order of device types is very import for creating the logical rank.
        # Users should make sure the order satisfies their needs. 
        self._hetero_device_types = args.hetero_device_types
        self._hetero_current_device_type = args.hetero_current_device_type
        self._rank_infos = {}
        self._physical_rank_to_logical_rank = {}
        self._logical_rank_to_physical_rank = {}
        self.build_rank_mapping()

    def build_rank_mapping(self):
        # Collect all rank infos.
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        all_rank_infos = [None] * world_size
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

    def to_physical_ranks(self, logical_ranks: list) -> list:
        """Converts logical ranks to physical ranks."""
        physical_ranks = []
        for logical_rank in logical_ranks:
            physical_ranks.append(
                self._logical_rank_to_physical_rank[logical_rank])
        return physical_ranks

    def to_logical_ranks(self, physical_ranks: list) -> list:
        """Converts physical ranks to logical ranks."""
        logical_ranks = []
        for physical_rank in physical_ranks:
            logical_ranks.append(
                self._physical_rank_to_logical_rank[physical_rank])
        return logical_ranks


class ProcessMesh:
    """ Define n-dimensional Cartesian process topology. """

    def __init__(
        self,
        data_parallel_size: int = 1,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        pipeline_model_parallel_split_rank: Optional[int] = None,
        use_sharp: bool = False,
        context_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        nccl_communicator_config_path: Optional[str] = None,
        distributed_timeout_minutes: int = 30,
        order: str = "tp-cp-ep-dp-pp",
        offset: int = 0,
        rank_mapper: RankMapper = None,
    ):
        assert torch.distributed.is_initialized()
        self._rank = torch.distributed.get_rank()
        self._world_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size * data_parallel_size 
        self._offset = offset

        if data_parallel_size % expert_model_parallel_size != 0:
            raise RuntimeError(
                f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
            )

        if expert_model_parallel_size > 1 and context_parallel_size > 1:
            raise RuntimeError(
                f"combination of expert model prallellism and context parallelism is not supported"
            )

        if virtual_pipeline_model_parallel_size is not None:
            if not pipeline_model_parallel_size > 2:
                raise RuntimeError(
                    "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
                )

            self._virtual_pipeline_model_parallel_rank = 0
            self._virtual_pipeline_model_parallel_world_size = virtual_pipeline_model_parallel_size

        self._pipeline_model_parallel_split_rank = pipeline_model_parallel_split_rank

        self._use_sharp = use_sharp

        self._nccl_comm_cfgs = {}
        if nccl_communicator_config_path is not None:
            try:
                import yaml
            except ImportError:
                raise RuntimeError(
                    "Cannot import `yaml`. Setting custom nccl communicator configs "
                    "requires the yaml package."
                )

            with open(nccl_communicator_config_path, "r") as stream:
                self._nccl_comm_cfgs = yaml.safe_load(stream)

        from megatron.core.parallel_state import RankGenerator 
        self._rank_generator = RankGenerator(
            tp=tensor_model_parallel_size,
            ep=expert_model_parallel_size,
            dp=data_parallel_size,
            pp=pipeline_model_parallel_size,
            cp=context_parallel_size,
            usp=1,
            order=order,
        )

        self._timeout = timedelta(minutes=distributed_timeout_minutes)
        self._rank_mapper = rank_mapper 
        self._group_ranks = {} # group_ranks belongs to the current rank 
        self._all_group_ranks = defaultdict(list) # group_ranks belongs to the current process mesh 
        self._process_groups = {} # process groups belongs to the current rank
        self._process_groups_gloo = {} # process groups belongs to the current rank with gloo backend

        self.build_all_process_groups()

    def build_process_group(
        self, token, independent_ep=False, gloo=False
    ):
        logical_ranks_list = self._rank_generator.get_ranks(token, independent_ep=independent_ep)
        # Add the offset for each ranks of the current process mesh
        for logical_ranks in logical_ranks_list:
            for i in range(len(logical_ranks)):
                logical_ranks[i] += self._offset 

        for logical_ranks in logical_ranks_list:
            group_name = self.get_group_name(token, independent_ep=independent_ep) 
            pg_options = get_nccl_options(group_name, self._nccl_comm_cfgs)
            ranks = self._rank_mapper.to_physical_ranks(logical_ranks)
            group = torch.distributed.new_group(
                ranks,
                timeout=self._timeout,
                pg_options=pg_options,
            )
            if gloo:
                group_gloo = torch.distributed.new_group(
                    ranks, timeout=self._timeout, backend="gloo"
                )
            self._all_group_ranks[group_name].append(ranks)
            if self._rank in ranks:
                self._group_ranks[group_name] = ranks
                self._process_groups[group_name] = group
                if gloo:
                    self._process_groups_gloo[group_name] = group_gloo

            # if token == "pp":
            #     if len(ranks) > 1:
            #         embedding_ranks = [ranks[0], ranks[-1]]
            #         position_embedding_ranks = [ranks[0]]
            #         if self._pipeline_model_parallel_split_rank is not None:
            #             if (
            #                 ranks[self._pipeline_model_parallel_split_rank]
            #                 not in embedding_ranks
            #             ):
            #                 embedding_ranks = [
            #                     ranks[0],
            #                     ranks[self._pipeline_model_parallel_split_rank],
            #                     ranks[-1],
            #                 ]
            #             if (
            #                 ranks[self._pipeline_model_parallel_split_rank]
            #                 not in position_embedding_ranks
            #             ):
            #                 position_embedding_ranks = [
            #                     ranks[0],
            #                     ranks[self._pipeline_model_parallel_split_rank],
            #                 ]
            #     else:
            #         embedding_ranks = ranks
            #         position_embedding_ranks = ranks

            #     self._all_group_ranks["embd"].append(embedding_ranks)
            #     group = torch.distributed.new_group(
            #         embedding_ranks,
            #         timeout=self._timeout,
            #         pg_options=get_nccl_options("embd", self._nccl_comm_cfgs),
            #     )
            #     if self._rank in embedding_ranks:
            #         self._process_groups["embd"] = group
            #     # TODO: need to check whether self._rank in ranks is correct
            #     # or should it be self._rank in embedding_ranks
            #     if self._rank in ranks:
            #         self._group_ranks["embd"] = embedding_ranks

            #     self._all_group_ranks["embd_pos"].append(position_embedding_ranks)
            #     group = torch.distributed.new_group(
            #         position_embedding_ranks,
            #         timeout=self.timeout,
            #         pg_options=get_nccl_options("embd", self._nccl_comm_cfgs),
            #     )
            #     if self._rank in position_embedding_ranks:
            #         self._process_groups["embd_pos"] = group
            #     # TODO: need to check whether self._rank in ranks is correct
            #     # or should it be self._rank in position_embedding_ranks
            #     if self._rank in ranks:
            #         self._group_ranks["embd_pos"] = position_embedding_ranks

        # if token == "pp":
        #     self._last_rank_when_using_pp = self._rank_mapper.to_physical_ranks(
        #         [logical_ranks_list[-1][-1]]
        #     )[0]

    def build_all_process_groups(self):
        self.build_process_group("dp", independent_ep=False, gloo=True)
        self.build_process_group("dp-cp", independent_ep=False, gloo=True)

        # Apply SHARP to DP process groups
        if self._use_sharp:
            if self._rank == 0:
                print(
                    "The number of process groups to use SHARP with depends on the type "
                    "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                    "process groups and QM2 supports up to 256 process groups. We apply "
                    "SHARP to the communications of the data-parallel domain. If the "
                    "number of data-parallel process groups is larger than the max "
                    "process groups that the network switch supports, the communication "
                    "will fall back to non-SHARP operators. To enable SHARP, "
                    "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
                )
            torch.distributed.barrier(
                group=self.get_process_group("dp-cp"),
                device_ids=[torch.cuda.current_device()],
            )
            # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
            os.environ["NCCL_COLLNET_ENABLE"] = "0"

        self.build_process_group("cp", independent_ep=False, gloo=False)
        self.build_process_group("tp-pp", independent_ep=False, gloo=False)
        self.build_process_group("tp-ep-pp", independent_ep=True, gloo=False)
        self.build_process_group('tp', independent_ep=False, gloo=False)
        self.build_process_group("pp", independent_ep=False, gloo=False)
        self.build_process_group("tp-dp-cp", independent_ep=False, gloo=False)
        self.build_process_group("tp-dp", independent_ep=False, gloo=False)
        self.build_process_group("tp-ep", independent_ep=True, gloo=False)
        self.build_process_group("ep", independent_ep=True, gloo=False)
        self.build_process_group("dp", independent_ep=True, gloo=True)
        self.build_process_group("dp-cp", independent_ep=False, gloo=True)
        self.build_process_group("dp-cp", independent_ep=True, gloo=True)
        self.build_process_group("usp", independent_ep=True, gloo=True)

    def get_parallel_size(self, token, independent_ep=False):
        if independent_ep:
            parallel_sizes = self._rank_generator.ordered_size_w_ep
            order = self._rank_generator.order_w_ep
        else:
            parallel_sizes = self._rank_generator.ordered_size_wo_ep
            order = self._rank_generator.order_wo_ep
        order = order.split("-")
        if token in order:
            return parallel_sizes[order.index(token)]
        else:
            raise ValueError(f"Invalid token: {token}")

    def get_group_name(self, token, independent_ep=False):
        names = {
            "tp-ep-pp": ("mp_exp", None),  # (with_ep, without_ep)
            "tp-ep": ("tp_exp", None),
            "ep": ("exp", None),
            "dp": ("dp_modulo_exp", "dp"),
            "dp-cp": ("dp_cp", "dp_modulo_exp_cp"), 
            "cp": ("cp", "cp"),
            "tp-pp": ("mp", "mp"),
            "tp": ("tp", "tp"),
            "pp": ("pp", "pp"),
            "tp-dp-cp": ("tp_dp_cp", "tp_dp_cp"),
            "tp-dp": ("tp_dp", "tp_dp"),
            "usp": ("usp", "usp"),
        }
        name_pair = names.get(token, None)
        if name_pair is None:
            raise ValueError(f"Invalid token: {token}")
        if independent_ep:
            name = name_pair[0]
            assert name is not None, f"Token {token} does not support independent ep."
            return name
        else:
            name = name_pair[1]
            assert name is not None, f"Token {token} does not support non-independent ep."
            return name

    def get_process_group(
        self,
        token,
        independent_ep=False,
        gloo=False,
        check_initialized=False,
    ):
        group_name = self.get_group_name(token, independent_ep=independent_ep)
        if gloo:
            group = self._process_groups_gloo.get(group_name, None)
        else:
            group = self._process_groups.get(group_name, None)
        if check_initialized:
            assert (
                group is not None
            ), f"Process group {group_name} is not initialized."
        return group

    def get_process_group_size(self, token, independent_ep=False, gloo=False):
        group_name = self.get_group_name(token, independent_ep=independent_ep)
        if gloo:
            return torch.distributed.get_world_size(self._process_groups_gloo[group_name])
        else:
            return torch.distributed.get_world_size(self._process_groups[group_name])

    def get_process_group_ranks(
        self, token, independent_ep=False, check_initialized=False
    ):
        group_name = self.get_group_name(token, independent_ep=independent_ep)
        ranks = self._group_ranks.get(group_name, None)
        if check_initialized:
            assert (
                ranks is not None
            ), f"Process group {group_name} is not initialized."
        return ranks

    def get_all_process_group_ranks(
        self, token, independent_ep=False, check_initialized=False
    ):
        group_name = self.get_group_name(token, independent_ep=independent_ep)
        ranks = self._all_group_ranks.get(group_name, None)
        if check_initialized:
            assert (
                ranks is not None
            ), f"Process group {group_name} is not initialized."
        return ranks

    def logical_coords_to_physical_ranks(self, coords, independent_ep=False):
        def _prefix_product(a: List[int], init=1) -> List[int]:
            r = [init]
            for v in a:
                init = init * v
                r.append(init)
            return r
        if independent_ep:
            for coord in coords:  
                assert len(coord) == 5
            sizes = self._rank_generator.ordered_size_w_ep
        else:
            for coord in coords:  
                assert len(coord) == 4
            sizes = self._rank_generator.ordered_size_wo_ep
        strides = _prefix_product(sizes)
        logical_ranks = []
        for coord in coords:  
            logical_rank = sum([c * s for c, s in zip(coord, strides)]) + self._offset
            logical_ranks.append(logical_rank)
        ranks = self._rank_mapper.to_physical_ranks(logical_ranks)
        return ranks


class ParallelContext:
    def __init__(self, args):
        assert torch.distributed.is_initialized()
        self._is_initialized = False 
        self._args = args
        self._current_process_mesh_index = 0 
        self._process_meshes = [] 
        self._rank_to_process_mesh = {}
        self._inter_mesh_group_ranks = defaultdict(list) 
        self._inter_mesh_process_groups_pp = {} # (src_rank, dst_rank) -> bool
        self._inter_mesh_process_groups_dp = {} # (src_rank, dst_rank) -> bool
        # (src_rank, local_tensor_shape, next) -> (dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size)
        self._inter_mesh_tensor_slices = {}
        self._group_ranks = defaultdict(list)
        self._all_group_ranks = defaultdict(list)
        self._process_groups = defaultdict(list) 
        self._process_group_to_ranks = {}
        self._parallel_world_sizes = {}
        self._parallel_ranks = {}
        self._timeout = timedelta(minutes=self._args.distributed_timeout_minutes)

        self._rank = torch.distributed.get_rank()
        self._rank_mapper = RankMapper(args)
        self.build_all_process_meshes()
        self.build_all_inter_mesh_process_groups()
        self.build_global_process_groups()
        from megatron.core.utils import GlobalMemoryBuffer
        self._global_memory_buffer = GlobalMemoryBuffer()

        self._is_initialized = True

    def is_initialized(self):
        return self._is_initialized

    def build_all_process_meshes(self):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        logical_rank = self._rank_mapper.to_logical_ranks([rank])[0]
        accumulated_world_size = 0
        for tp, cp, ep, dp, pp in self._args.hetero_process_meshes:
            process_mesh = ProcessMesh(
                tensor_model_parallel_size=tp,
                context_parallel_size=cp,
                data_parallel_size=dp,
                pipeline_model_parallel_size=pp,
                expert_model_parallel_size=ep,
                nccl_communicator_config_path=self._args.nccl_communicator_config_path,
                distributed_timeout_minutes=self._args.distributed_timeout_minutes,
                order='tp-usp-cp-ep-dp-pp' if not self._args.use_tp_pp_dp_mapping else 'tp-pp-dp',
                offset=accumulated_world_size,
                rank_mapper=self._rank_mapper,
            )
            if (
                logical_rank >= accumulated_world_size
                and logical_rank < accumulated_world_size + process_mesh._world_size
            ):
                self._current_process_mesh_index = len(self._process_meshes)
            accumulated_world_size += process_mesh._world_size
            self._process_meshes.append(process_mesh)
        if world_size != accumulated_world_size:
            raise RuntimeError(
                f"World size mismatch. Expected {world_size}, but got {accumulated_world_size}"
            )

        all_rank_to_process_mesh = [None for _ in range(world_size)]
        cur_rank_to_process_mesh = {
            "rank": rank,
            "process_mesh_idx": self._current_process_mesh_index,
        }
        torch.distributed.all_gather_object(
            all_rank_to_process_mesh, cur_rank_to_process_mesh
        )
        for item in all_rank_to_process_mesh:
            self._rank_to_process_mesh[item["rank"]] = self._process_meshes[
                item["process_mesh_idx"]
            ]

    def build_inter_mesh_process_groups(self, process_mesh1, process_mesh2):
        tp1 = process_mesh1.get_parallel_size("tp", independent_ep=False)
        cp1 = process_mesh1.get_parallel_size("cp", independent_ep=False)
        dp1 = process_mesh1.get_parallel_size("dp", independent_ep=False)
        tp2 = process_mesh2.get_parallel_size("tp", independent_ep=False)
        cp2 = process_mesh2.get_parallel_size("cp", independent_ep=False)
        dp2 = process_mesh2.get_parallel_size("dp", independent_ep=False)

        if not(tp1 == 1 and tp2 == 1):
            sp1 = tp1 * cp1
            sp2 = tp2 * cp2
        else:
            sp1 = cp1
            sp2 = cp2
        sp_overlapped_mapping = find_overlapped_mapping(sp1, sp2)
        dp_overlapped_mapping = find_overlapped_mapping(dp1, dp2)
        src_pp_dims = [process_mesh1.get_parallel_size("pp") - 1]
        dst_pp_dims = [0]
        # i is tp, j is cp, k is dp, 
        for s in range(sp1):
            src_i, src_j = s % tp1, s // tp1
            finded_mp_group = False
            for k in range(dp1):
                src_coord = [src_i, src_j, k, src_pp_dims[0]]
                dst_sp_dims = [dim for dim, _, _ in sp_overlapped_mapping[s]]
                dst_dp_dims = [dim for dim, _, _ in dp_overlapped_mapping[k]]
                dst_coords = list(
                    itertools.product(dst_sp_dims, dst_dp_dims, dst_pp_dims)
                )
                src_rank = process_mesh1.logical_coords_to_physical_ranks(
                    [src_coord]
                )[0]
                # find pp group connection
                for dst_coord in dst_coords:
                    sp_dim, dp_dim, pp_dim = dst_coord
                    dst_coord = [sp_dim % tp2, sp_dim // tp2, dp_dim, pp_dim]
                    dst_rank = process_mesh2.logical_coords_to_physical_ranks(
                        [dst_coord]
                    )[0]
                    # NOTE: There is no need to create a group for the commnetting boundary.
                    #       We will create the `pp` group in the `build_global_process_groups` function.
                    # ranks = [src_rank, dst_rank]
                    # timeout = max(process_mesh1._timeout, process_mesh2._timeout)
                    # group = torch.distributed.new_group(ranks, timeout=timeout)
                    self._inter_mesh_process_groups_pp[(src_rank, dst_rank)] = True
                
                # find mp(tp+pp) group connection
                if not finded_mp_group:
                    finded_mp_group = True
                    for k in range(dp1):
                        src_coord = [tp1 - 1, cp1 - 1, k, src_pp_dims[0]]
                        dst_dp_dims = [dim for dim, _, _ in dp_overlapped_mapping[k]]
                        dst_coords = list(
                            itertools.product([0], [0], dst_dp_dims, dst_pp_dims)
                        )
                        src_rank = process_mesh1.logical_coords_to_physical_ranks(
                            [src_coord]
                        )[0]
                        for dst_coord in dst_coords:
                            tp_dim, cp_dim, dp_dim, pp_dim = dst_coord
                            dst_coord = [tp_dim, cp_dim, dp_dim, pp_dim]
                            dst_rank = process_mesh2.logical_coords_to_physical_ranks(
                                [dst_coord]
                            )[0]
                            self._inter_mesh_process_groups_dp[(src_rank, dst_rank)] = True
                

    def build_all_inter_mesh_process_groups(self):
        if len(self._process_meshes) == 1:
            return

        for i in range(len(self._process_meshes) - 1):
            self.build_inter_mesh_process_groups(
                self._process_meshes[i], self._process_meshes[i + 1]
            )

    def build_global_process_groups(self):
        # build global pipeline process groups
        def _backtrack(mesh_index, prev_rank, path, token = "pp", independent_ep=False):
            group_name = self._process_meshes[0].get_group_name(token, independent_ep=independent_ep)
            if mesh_index == len(self._process_meshes):
                aggregated_ranks = [rank for ranks in path for rank in ranks]
                self._all_group_ranks[group_name].append(aggregated_ranks)
                group = torch.distributed.new_group(aggregated_ranks, timeout=self._timeout)
                if self._rank in aggregated_ranks:
                    self._process_groups[group_name].append(group)
                    self._group_ranks[group_name].append(aggregated_ranks)
                    self._process_group_to_ranks[group] = aggregated_ranks
                return
            current_mesh = self._process_meshes[mesh_index]
            ranks_list = current_mesh.get_all_process_group_ranks(token, independent_ep=independent_ep, check_initialized=True)
            valid_ranks_list = []
            for ranks in ranks_list:
                mesh_is_connected = False
                for prev_path_ranks in path:
                    for prev_path_rank in prev_path_ranks:
                        if token == "pp" and (prev_path_rank, ranks[0]) in self._inter_mesh_process_groups_pp:
                            mesh_is_connected = True
                        elif token == "tp-pp" and (prev_path_rank, ranks[0]) in self._inter_mesh_process_groups_dp:
                            mesh_is_connected = True
                if prev_rank == -1 or mesh_is_connected:
                    valid_ranks_list.append(ranks)
            for ranks in valid_ranks_list:
                path.append(ranks)
                _backtrack(mesh_index + 1, ranks[-1], path, token=token, independent_ep=independent_ep)
                path.pop()

        for mesh_index, process_mesh in enumerate(self._process_meshes):
            ranks_list = process_mesh.get_all_process_group_ranks(
                "tp-ep-pp", independent_ep=True, check_initialized=True
            )
            ranks = list(itertools.chain.from_iterable(ranks_list))
            self._all_group_ranks["mp_exp"].append(ranks)
            group = torch.distributed.new_group(ranks, timeout=self._timeout)
            if self._rank in ranks:
                self._group_ranks["mp_exp"] = ranks
                self._process_groups["mp_exp"] = group
                self._process_group_to_ranks[group] = ranks
            ranks_list = process_mesh.get_all_process_group_ranks(
                "pp", independent_ep=False, check_initialized=True
            )
            ranks = list(itertools.chain.from_iterable(ranks_list))
            if "last_rank" not in self._parallel_ranks:
                self._parallel_ranks["last_rank"] = []
            self._parallel_ranks["last_rank"].append(ranks[-1])
        _backtrack(0, -1, path=[], token="tp-pp", independent_ep=False)
        _backtrack(0, -1, path=[], token="pp", independent_ep=False)
        # build global embedding process groups
        for ranks in self._group_ranks["pp"]:
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
                # `pp_split` is similar to the `pipeline_model_parallel_split_rank` in parallel_state
                if "pp_split" in self._parallel_ranks.keys() and self._parallel_ranks["pp_split"] is not None:
                    split_rank = self._parallel_ranks["pp_split"]
                    if ranks[split_rank] not in embedding_ranks:
                        embedding_ranks = [
                            ranks[0],
                            ranks[split_rank],
                            ranks[-1],
                        ]
                    if ranks[split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [
                            ranks[0],
                            ranks[split_rank],
                        ]
            else:
                embedding_ranks = ranks
                position_embedding_ranks = ranks
            group = torch.distributed.new_group(
                embedding_ranks, timeout=self._timeout
            )
            if self._rank in embedding_ranks:
                self._process_groups["embd"].append(group)
                self._process_group_to_ranks[group] = embedding_ranks
            
            if self._rank in ranks:
                self._group_ranks["embd"].append(embedding_ranks)

            group = torch.distributed.new_group(
                position_embedding_ranks, timeout=self._timeout
            )
            if self._rank in position_embedding_ranks:
                self._process_groups["embd_pos"].append(group)
                self._process_group_to_ranks[group] = position_embedding_ranks

            if self._rank in ranks:
                self._group_ranks["embd_pos"].append(position_embedding_ranks)

    def get_inter_mesh_process_group(self, src_rank, dst_rank):
        if (src_rank, dst_rank) in self._inter_mesh_process_groups_pp:
            return self._inter_mesh_process_groups_pp[(src_rank, dst_rank)]
        elif (dst_rank, src_rank) in self._inter_mesh_process_groups_pp:
            return self._inter_mesh_process_groups_pp[(dst_rank, src_rank)]
        else:
            raise RuntimeError(
                f"ProcessGroup [{src_rank}, {dst_rank}] does not exist."
            )

    def get_inter_mesh_tensor_slices(self, rank, local_tensor_shape, next=True):
        if (rank, local_tensor_shape, next) in self._inter_mesh_tensor_slices:
            return self._inter_mesh_tensor_slices[(rank, local_tensor_shape, next)]
        process_mesh1 = self._process_meshes[self._current_process_mesh_index]
        if next:
            process_mesh2 = self.get_next_process_mesh()
            # first stage of the next process mesh
            src_pp_dims = [process_mesh1.get_parallel_size("pp") - 1]
            dst_pp_dims = [0]
        else:
            process_mesh2 = self.get_prev_process_mesh()
            # last stage of the previous process mesh
            src_pp_dims = [0]
            dst_pp_dims = [process_mesh2.get_parallel_size("pp") - 1]
        tp1 = process_mesh1.get_parallel_size("tp", independent_ep=False)
        cp1 = process_mesh1.get_parallel_size("cp", independent_ep=False)
        dp1 = process_mesh1.get_parallel_size("dp", independent_ep=False)
        tp2 = process_mesh2.get_parallel_size("tp", independent_ep=False)
        cp2 = process_mesh2.get_parallel_size("cp", independent_ep=False)
        dp2 = process_mesh2.get_parallel_size("dp", independent_ep=False)

        # Assume that the tensor shape is (seq_len, batch_size, hidden_size)
        local_seq_len, local_batch_size, local_hidden_size = local_tensor_shape
        if not(tp1 == 1 and tp2 == 1):
            global_seq_len = local_seq_len * tp1 * cp1
            sp1 = tp1 * cp1
            sp2 = tp2 * cp2
        else:
            global_seq_len = local_seq_len * cp1 
            sp1 = cp1
            sp2 = cp2
        global_batch_size = local_batch_size * dp1
        sp_overlapped_mapping = find_overlapped_mapping(sp1, sp2, global_seq_len)
        dp_overlapped_mapping = find_overlapped_mapping(dp1, dp2, global_batch_size)
        for s in range(sp1):
            src_i, src_j = s % tp1, s // tp1
            for k in range(dp1):
                src_coord = [src_i, src_j, k, src_pp_dims[0]]
                dst_sp_dims = [c for c, _, _ in sp_overlapped_mapping[s]]
                dst_dp_dims = [c for c, _, _ in dp_overlapped_mapping[k]]
                dst_coords = list(
                    itertools.product(dst_sp_dims, dst_dp_dims, dst_pp_dims)
                )
                src_sp_starts = [s for _, s, _ in sp_overlapped_mapping[s]]
                src_dp_starts = [s for _, s, _ in dp_overlapped_mapping[k]]
                src_starts = list(itertools.product(src_sp_starts, src_dp_starts))
                src_sp_ends = [e for _, _, e in sp_overlapped_mapping[s]]
                src_dp_ends = [e for _, _, e in dp_overlapped_mapping[k]]
                src_ends = list(itertools.product(src_sp_ends, src_dp_ends))
                src_rank = process_mesh1.logical_coords_to_physical_ranks([src_coord])[0]
                for i, dst_coord in enumerate(dst_coords):
                    sp_dim, dp_dim, pp_dim = dst_coord
                    dst_coord = [sp_dim % tp2, sp_dim // tp2, dp_dim, pp_dim]
                    dst_rank = process_mesh2.logical_coords_to_physical_ranks([dst_coord])[0]
                    sp_start, dp_start = src_starts[i]
                    sp_end, dp_end = src_ends[i]
                    if (src_rank, local_tensor_shape, next) not in self._inter_mesh_tensor_slices: 
                        self._inter_mesh_tensor_slices[(src_rank, local_tensor_shape, next)] = []
                    self._inter_mesh_tensor_slices[
                        (src_rank, local_tensor_shape, next)
                    ].append(
                        (
                            dst_rank,
                            (dp_start, dp_end),
                            (sp_start, sp_end),
                            local_hidden_size,
                        )
                    )
        return self._inter_mesh_tensor_slices[(rank, local_tensor_shape, next)]

    def get_current_process_mesh(self):
        assert self._current_process_mesh_index < len(self._process_meshes)
        return self._process_meshes[self._current_process_mesh_index]

    def get_prev_process_mesh(self):
        assert self._current_process_mesh_index - 1 >= 0 
        return self._process_meshes[self._current_process_mesh_index - 1]

    def get_next_process_mesh(self):
        assert self._current_process_mesh_index + 1 < len(self._process_meshes)
        return self._process_meshes[self._current_process_mesh_index + 1]

    def get_model_parallel_group(self, with_expert_parallel=False):
        """Get the model parallel group the caller rank belongs to."""
        if with_expert_parallel:
            group = self._process_groups.get("mp_exp", None)
            assert group is not None, "model parallel group is not initialized"
            return group 
        group = self._process_groups.get("mp", None)
        assert group is not None, 'model parallel group is not initialized'
        return group

    def get_tensor_model_parallel_group(self, check_initialized=True):
        """Get the tensor model parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "tp", independent_ep=False, gloo=False, check_initialized=check_initialized
        )

    def get_pipeline_model_parallel_group(self, check_initialized=True):
        """Get the pipeline model parallel group the caller rank belongs to."""
        group = self._process_groups.get("pp", None)
        assert group is not None, "pipeline_model parallel group is not initialized"
        return self._process_groups["pp"]

    def get_data_parallel_group(self, with_context_parallel=False):
        """Get the data parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            return current_process_mesh.get_process_group(
                "dp-cp", independent_ep=False, gloo=False, check_initialized=True
            )
        else:
            return current_process_mesh.get_process_group(
                "dp", independent_ep=False, gloo=False, check_initialized=True
            )

    def get_data_parallel_group_gloo(self, with_context_parallel=False):
        """Get the data parallel group-gloo the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            return current_process_mesh.get_process_group(
                "dp-cp", independent_ep=False, gloo=True, check_initialized=True
            )
        else:
            return current_process_mesh.get_process_group(
                "dp", independent_ep=False, gloo=True, check_initialized=True
            )

    def get_context_parallel_group(self, check_initialized=True):
        """Get the context parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "cp", independent_ep=False, gloo=False, check_initialized=check_initialized
        )

    def get_context_parallel_global_ranks(self, check_initialized=True):
        """Get all global ranks of the context parallel group that the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group_ranks(
            "cp", independent_ep=False, check_initialized=check_initialized
        )
    
    def get_ulysses_sp_parallel_group(self, check_initialized=True):
        """Get the ulysses sequence parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "usp", independent_ep=False, gloo=False, check_initialized=check_initialized
        )
        
    def get_ulysses_sp_parallel_global_ranks(self, check_initialized=True):
        """Get all global ranks of the ulysses sequence parallel group that the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group_ranks(
            "usp", independent_ep=False, check_initialized=check_initialized
        )

    def get_embedding_group(self):
        """Get the embedding group the caller rank belongs to."""
        groups = self._process_groups.get("embd", None)
        assert groups is not None, 'embedding group is not initialized'
        for group in groups:
            if self._rank in self._process_group_to_ranks[group]:
                embd_group = group
                break
        return embd_group 

    def get_position_embedding_group(self):
        """Get the position embedding group the caller rank belongs to."""
        groups = self._process_groups.get("embd_pos", None)
        assert groups is not None, 'Position embedding group is not initialized'
        for group in groups:
            if self._rank in self._process_group_to_ranks[group]:
                pos_embd_group = group
                break
        return pos_embd_group

    def get_amax_reduction_group(self, with_context_parallel=False):
        """Get the FP8 amax reduction group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            return current_process_mesh.get_process_group(
                "tp-dp-cp", independent_ep=False, gloo=False, check_initialized=True
            )
        else:
            return current_process_mesh.get_process_group(
                "tp-dp", independent_ep=False, gloo=False, check_initialized=True
            )

    def get_tensor_and_data_parallel_group(self, with_context_parallel=False):
        """Get the tensor and data parallel group the caller rank belongs to."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            return current_process_mesh.get_process_group(
                "tp-dp-cp", independent_ep=False, gloo=False, check_initialized=True
            )
        else:
            return current_process_mesh.get_process_group(
                "tp-dp", independent_ep=False, gloo=False, check_initialized=True
            )

    def get_expert_model_parallel_group(self):
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "ep", independent_ep=True, gloo=False, check_initialized=True
        )

    def get_tensor_and_expert_parallel_group(self):
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "tp-ep", independent_ep=True, gloo=False, check_initialized=True
        )

    def get_data_modulo_expert_parallel_group(self, with_context_parallel=False):
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            return current_process_mesh.get_process_group(
                "dp-cp", independent_ep=True, gloo=False, check_initialized=True
            )
        else:
            
            return current_process_mesh.get_process_group(
                "dp", independent_ep=True, gloo=False, check_initialized=True
            )

    def get_data_modulo_expert_parallel_group_gloo(self):
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        return current_process_mesh.get_process_group(
            "dp", independent_ep=True, gloo=True, check_initialized=True
        )

    def set_expert_model_parallel_world_size(self, world_size):
        self._parallel_world_sizes["ep"] = world_size

    def set_tensor_model_parallel_world_size(self, world_size):
        """Set the tensor model parallel size"""
        self._parallel_world_sizes["tp"] = world_size

    def set_pipeline_model_parallel_world_size(self, world_size):
        """Set the pipeline model parallel size"""
        self._parallel_world_sizes["pp"] = world_size

    def set_virtual_pipeline_model_parallel_world_size(self, world_size):
        """Set the pipeline model parallel size"""
        self._parallel_world_sizes["vpp"] = world_size

    def get_tensor_model_parallel_world_size(self):
        """Return world size for the tensor model parallel group."""
        size = self._parallel_world_sizes.get("tp", None)
        if size is not None:
            return size
        return torch.distributed.get_world_size(group=self.get_tensor_model_parallel_group())

    def get_pipeline_model_parallel_world_size(self, group=None):
        """Return world size for the pipeline model parallel group."""
        size = self._parallel_world_sizes.get("pp", None)
        if size is not None:
            return size
        if group is None:
            group = self.get_pipeline_model_parallel_group()[0]
        return torch.distributed.get_world_size(group)

    def set_expert_model_parallel_rank(self, rank):
        """Set expert model parallel rank."""
        self._parallel_ranks["ep"] = rank

    def set_tensor_model_parallel_rank(self, rank):
        """Set tensor model parallel rank."""
        self._parallel_ranks["tp"] = rank

    def set_pipeline_model_parallel_rank(self, rank):
        """Set pipeline model parallel rank."""
        self._parallel_ranks["pp"] = rank

    def set_pipeline_model_parallel_split_rank(self, rank):
        """Set pipeline model parallel split rank."""
        self._parallel_ranks["pp-split"] = rank

    def get_tensor_model_parallel_rank(self):
        """Return my rank for the tensor model parallel group."""
        rank = self._parallel_ranks.get("tp", None)
        if rank is not None:
            return rank
        return torch.distributed.get_rank(group=self.get_tensor_model_parallel_group())

    def get_pipeline_model_parallel_rank(self, group=None):
        """Return my rank for the pipeline model parallel group."""
        rank = self._parallel_ranks.get("pp", None)
        if rank is not None:
            return rank
        if group is None:
            group = self.get_pipeline_model_parallel_group()[0]
        return torch.distributed.get_rank(group=group)

    def get_pipeline_model_parallel_split_rank(self):
        """Return pipeline model parallel split rank."""
        return self._parallel_ranks.get("pp-split", None)

    def is_pipeline_first_stage(self, ignore_virtual=False, group=None):
        """Return True if in the first pipeline model-parallel stage, False otherwise."""
        if not ignore_virtual:
            if (
                self.get_virtual_pipeline_model_parallel_world_size() is not None
                and self.get_virtual_pipeline_model_parallel_rank() != 0
            ):
                return False
        return self.get_pipeline_model_parallel_rank(group) == 0

    def is_pipeline_last_stage(self, ignore_virtual=False, group=None):
        """Return True if in the last pipeline model-parallel stage, False otherwise."""
        if not ignore_virtual:
            virtual_pipeline_model_parallel_world_size = (
                self.get_virtual_pipeline_model_parallel_world_size()
            )
            if (
                virtual_pipeline_model_parallel_world_size is not None
                and self.get_virtual_pipeline_model_parallel_rank()
                != (virtual_pipeline_model_parallel_world_size - 1)
            ):
                return False
        return self.get_pipeline_model_parallel_rank(group) == (self.get_pipeline_model_parallel_world_size(group) - 1)

    def is_rank_in_embedding_group(self, ignore_virtual=False, group=None):
        """Return true if current rank is in embedding group, False otherwise."""
        rank = torch.distributed.get_rank()
        if group is None:
            group = self._process_groups.get("embd", None)
            if group is None:
                return False
            else:
                group = group[0]
        ranks = self._process_group_to_ranks[group]
        if ignore_virtual:
            return rank in ranks 
        if rank in ranks:
            if rank == ranks[0]:
                return self.is_pipeline_first_stage(ignore_virtual=False, group=group)
            elif rank == ranks[-1]:
                return self.is_pipeline_last_stage(ignore_virtual=False, group=group)
            else:
                return True
        return False

    def is_rank_in_position_embedding_group(self, group=None):
        """Return true if current rank is in position embedding group, False otherwise."""
        rank = torch.distributed.get_rank()
        if group is None:
            group = self._process_groups.get("embd_pos", None)
            if group is None:
                return False
            else:
                group = group[0]
        ranks = self._process_group_to_ranks[group]
        return rank in ranks 

    def is_pipeline_stage_before_split(self, rank=None, group=None):
        """Return True if pipeline stage executes encoder block for a model
        with both encoder and decoder."""
        if self.get_pipeline_model_parallel_world_size(group) == 1:
            return True
        if rank is None:
            rank = self.get_pipeline_model_parallel_rank(group)
        split_rank = self.get_pipeline_model_parallel_split_rank()
        if split_rank is None:
            return True
        if rank < split_rank:
            return True
        return False

    def is_pipeline_stage_after_split(self, rank=None, group=None):
        """Return True if pipeline stage executes decoder block for a model
        with both encoder and decoder."""
        if self.get_pipeline_model_parallel_world_size(group) == 1:
            return True
        if rank is None:
            rank = self.get_pipeline_model_parallel_rank(group)
        split_rank = self.get_pipeline_model_parallel_split_rank()
        if split_rank is None:
            return True
        if rank >= split_rank:
            return True
        return False

    def is_pipeline_stage_at_split(self, group=None):
        """Return true if pipeline stage executes decoder block and next
        stage executes encoder block for a model with both encoder and
        decoder."""
        rank = self.get_pipeline_model_parallel_rank(group)
        return self.is_pipeline_stage_before_split(rank, group) and self.is_pipeline_stage_after_split(rank + 1, group)

    def get_virtual_pipeline_model_parallel_rank(self):
        """Return the virtual pipeline-parallel rank."""
        return self._parallel_ranks.get("vpp", None)

    def set_virtual_pipeline_model_parallel_rank(self, rank):
        """Set the virtual pipeline-parallel rank."""
        self._parallel_ranks["vpp"] = rank

    def get_virtual_pipeline_model_parallel_world_size(self):
        """Return the virtual pipeline-parallel world size."""
        return self._parallel_world_sizes.get("vpp", None)

    def get_tensor_model_parallel_src_rank(self):
        """Calculate the global rank corresponding to the first local rank
        in the tensor model parallel group."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        ranks = current_process_mesh.get_process_group_ranks(
            "tp", independent_ep=False, check_initialized=True
        )
        return ranks[0]

    def get_data_parallel_src_rank(self, with_context_parallel=False):
        """Calculate the global rank corresponding to the first local rank
        in the data parallel group."""
        current_process_mesh = self._process_meshes[self._current_process_mesh_index]
        if with_context_parallel:
            ranks = current_process_mesh.get_process_group_ranks(
                "dp-cp", independent_ep=False, check_initialized=True
            )
        else:
            ranks = current_process_mesh.get_process_group_ranks(
                "dp", independent_ep=False, check_initialized=True
            )
        return ranks[0]

    def get_pipeline_model_parallel_first_rank(self, group=None):
        """Return the global rank of the first process in the pipeline for the
        current tensor parallel group"""
        if group is None:
            group = self.get_pipeline_model_parallel_group()[0]
        ranks = self._process_group_to_ranks.get(group, None)
        assert ranks is not None, "Pipeline parallel group is not initialized"
        return ranks[0]

    def get_pipeline_model_parallel_last_rank(self, group=None):
        """Return the global rank of the last process in the pipeline for the
        current tensor parallel group"""
        if group is None:
            group = self.get_pipeline_model_parallel_group()[0]
        ranks = self._process_group_to_ranks.get(group, None)
        assert ranks is not None, "Pipeline parallel group is not initialized"
        last_rank_local = self.get_pipeline_model_parallel_world_size(group) - 1
        return ranks[last_rank_local]

    def get_pipeline_model_parallel_next_rank(self, group=None):
        """Return the global rank that follows the caller in the pipeline"""
        if group is None:
            group = self.get_pipeline_model_parallel_group()[0]
        ranks = self._process_group_to_ranks.get(group, None)
        assert ranks is not None, "Pipeline parallel group is not initialized"
        rank_in_pipeline = self.get_pipeline_model_parallel_rank(group)
        world_size = self.get_pipeline_model_parallel_world_size(group)
        return ranks[(rank_in_pipeline + 1) % world_size]

    def get_pipeline_model_parallel_prev_rank(self, group=None):
        """Return the global rank that preceeds the caller in the pipeline"""
        if group is None:
            group = self.get_pipeline_model_parallel_group()[0]
        ranks = self._process_group_to_ranks.get(group, None)
        assert ranks is not None, "Pipeline parallel group is not initialized"
        rank_in_pipeline = self.get_pipeline_model_parallel_rank(group)
        world_size = self.get_pipeline_model_parallel_world_size(group)
        return ranks[(rank_in_pipeline - 1) % world_size]

    def get_last_rank_when_using_pipeline(self):
        """Return the global rank of the last process in the pipeline"""
        assert (
            self._parallel_ranks.get("last_rank", None) is not None
        ), "Last rank when using pipeline is not initialized"
        return self._parallel_ranks["last_rank"][self._current_process_mesh_index]

    def get_data_parallel_world_size(self, with_context_parallel=False):
        """Return world size for the data parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(
                group=self.get_data_parallel_group(with_context_parallel=with_context_parallel)
            )
        else:
            return 0

    def get_data_parallel_rank(self, with_context_parallel=False):
        """Return my rank for the data parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(
                group=self.get_data_parallel_group(with_context_parallel=with_context_parallel)
            )
        else:
            return 0

    def get_context_parallel_world_size(self):
        """Return world size for the context parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(group=self.get_context_parallel_group())
        else:
            return 0

    def get_context_parallel_rank(self):
        """Return my rank for the context parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(group=self.get_context_parallel_group())
        else:
            return 0
    
    def get_ulysses_sp_parallel_world_size(self):
        """Return world size for the ulysses sequence parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(group=self.get_ulysses_sp_parallel_group())
        else:
            return 0

    def get_ulysses_sp_parallel_rank(self):
        """Return my rank for the ulysses sequence parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(group=self.get_ulysses_sp_parallel_group())
        else:
            return 0

    def get_expert_model_parallel_world_size(self):
        """Return world size for the expert model parallel group"""
        if self._parallel_world_sizes.get("ep", None) is not None:
            return self._parallel_world_sizes["ep"]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
                group=self.get_tensor_and_expert_parallel_group()
            )
            return tensor_and_expert_parallel_world_size // self.get_tensor_model_parallel_world_size()
        else:
            return 0

    def get_tensor_and_expert_parallel_world_size(self):
        """Return world size for the expert model parallel group times model parallel group.
           Currently, each expert will also be distributed across TP group by default.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
                group=self.get_tensor_and_expert_parallel_group()
            )
            return tensor_and_expert_parallel_world_size
        else:
            return 0

    def get_expert_model_parallel_rank(self):
        """Return my rank for the expert parallel group"""
        if self._parallel_ranks.get("ep", None) is not None:
            return self._parallel_ranks["ep"]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tensor_and_expert_parallel_rank = torch.distributed.get_rank(
                group=self.get_tensor_and_expert_parallel_group()
            )
            return tensor_and_expert_parallel_rank // self.get_tensor_model_parallel_world_size()
        else:
            return 0

    def get_data_modulo_expert_parallel_rank(self, with_context_parallel):
        """Return my rank for the context parallel group."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(group=self.get_data_modulo_expert_parallel_group(with_context_parallel=with_context_parallel))
        else:
            return 0

    def get_tensor_and_expert_parallel_rank(self):
        """Return my rank for the tensor and expert parallel group"""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(group=self.get_tensor_and_expert_parallel_group())
        else:
            return 0

    def set_global_memory_buffer(self):
        """Initialize global buffer"""
        assert self._global_memory_buffer is None, 'global memory buffer is already initialized'
        self._global_memory_buffer = GlobalMemoryBuffer()

    def get_global_memory_buffer(self):
        """Return the global GlobalMemoryBuffer object"""
        assert self._global_memory_buffer is not None, 'global memory buffer is not initialized'
        return self._global_memory_buffer

    def destroy_global_memory_buffer(self):
        """Sets the global memory buffer to None"""
        self._global_memory_buffer = None
