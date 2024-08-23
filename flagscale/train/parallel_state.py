# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import os
import warnings
from datetime import timedelta
from functools import partial
from itertools import cycle
from typing import Callable, List, Optional

import torch

from megatron.core.utils import GlobalMemoryBuffer
from megatron.core import parallel_state as PS
from megatron.core.parallel_state import RankGenerator

from flagscale.train import get_parallel_context  

# Ulysses sequence parallel group that the current rank belongs to
_ULYSSES_SP_PARALLEL_GROUP = None
# A list of global ranks for each ulysses sequence parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between ulysses suquence parallel_ranks
_ULYSSES_SP_PARALLEL_GLOBAL_RANKS = None

# Data parallel group information with ulysses sequence parallel combined.
_DATA_PARALLEL_GROUP_WITH_USP = None
_DATA_PARALLEL_GROUP_WITH_USP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_USP = None

# Data parallel group information with ulysses sequence and context parallel combined.
_DATA_PARALLEL_GROUP_WITH_USP_CP = None
_DATA_PARALLEL_GROUP_WITH_USP_CP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_USP_CP = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    ulysses_parallel_size: int  = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-usp-cp-ep-dp-pp",
    encoder_tensor_model_parallel_size: Optional[int] = 0,
    encoder_pipeline_model_parallel_size: Optional[int] = 0,
    get_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
    get_position_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
) -> None:
    """Initialize model data parallel groups.

    Args:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            DEPRECATED. For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        use_sharp (bool, default = False):
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.
        
        ulysses_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across using deepseed-ulysses method.

        expert_model_parallel_size (int, default = 1):
            The number of Mixture of Experts parallel GPUs in each expert
            parallel group.

        nccl_communicator_config_path (str, default = None):
            Path to the yaml file of NCCL communicator configurations.
            `min_ctas`, `max_ctas`, and `cga_cluster_size` can be set
            for each communicator.

        distributed_timeout_minutes (int, default = 30): Timeout, in
            minutes,for operations executed against distributed
            process groups. See PyTorch documentation at
            https://pytorch.org/docs/stable/distributed.html for
            caveats.

        order (str, default=tp-dp-pp):
            The rank initialization order of parallelism. Now we support
            tp-dp-pp and tp-pp-dp orders.

        encoder_tensor_model_parallel_size (int, default = 0):
            The number of GPUs to split individual tensors across in the encoder. If 0,
            then we use the default, decoder's tensor model parallel size.

        encoder_pipeline_model_parallel_size (int, default = 0):
            The number of tensor parallel GPU groups to allocate to the encoder. As an example,
            if pipeline_model_parallel_size is 4 and encoder_pipeline_model_parallel_size is 2,
            then the encoder will use the first two pipeline stages for its layers, and the total
            amount of pipelineing is 6.

        get_embedding_ranks (Callable[[List[int], Optional[int]], List[int]], optional, default=None):
            A function that takes in a list of ranks for a pipeline group and returns
            those ranks that should have embeddings.

        get_position_embedding_ranks (Callable[[List[int], Optional[int]], List[int]], optional, default=None):
            A function that takes in a list of ranks for a pipeline group, and returns
            those ranks that should have position embeddings.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
    PS.initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank,
        use_sharp,
        ulysses_parallel_size,
        context_parallel_size,
        expert_model_parallel_size,
        nccl_communicator_config_path,
        distributed_timeout_minutes,
        order,
        encoder_tensor_model_parallel_size,
        encoder_pipeline_model_parallel_size,
        get_embedding_ranks,
        get_position_embedding_ranks
        )
    
    initialize_model_parallel_for_usp(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        use_sharp,
        ulysses_parallel_size,
        context_parallel_size,
        expert_model_parallel_size,
        nccl_communicator_config_path,
        distributed_timeout_minutes,
        order,
        encoder_tensor_model_parallel_size,
        encoder_pipeline_model_parallel_size,
    )
    

def initialize_model_parallel_for_usp(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    use_sharp: bool = False,
    ulysses_parallel_size: int  = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-usp-cp-ep-dp-pp",
    encoder_tensor_model_parallel_size: Optional[int] = 0,
    encoder_pipeline_model_parallel_size: Optional[int] = 0,
) -> None:

    # Get world size and rank. Ensure some consistencies.
    world_size: int = torch.distributed.get_world_size()

    encoder_model_size = (
        encoder_tensor_model_parallel_size
        * encoder_pipeline_model_parallel_size
        * context_parallel_size
        * ulysses_parallel_size
    )
    decoder_model_size = (
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size * ulysses_parallel_size
    )
    total_model_size = encoder_model_size + decoder_model_size
    data_parallel_size: int = world_size // total_model_size
    encoder_world_size = encoder_model_size * data_parallel_size
    decoder_world_size = decoder_model_size * data_parallel_size

    assert (
        encoder_world_size + decoder_world_size == world_size
    ), f"{encoder_world_size=} + {decoder_world_size=} != {world_size=}"

    if encoder_world_size > 0:
        encoder_rank_generator = RankGenerator(
            tp=encoder_tensor_model_parallel_size,
            ep=1,
            dp=data_parallel_size,
            pp=encoder_pipeline_model_parallel_size,
            cp=context_parallel_size,
            usp=1,
            order=order,
            rank_offset=0,
        )
    else:
        encoder_rank_generator = None

    decoder_rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=expert_model_parallel_size,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        usp=ulysses_parallel_size,
        order=order,
        rank_offset=encoder_world_size,
    )

    def generator_wrapper(group_type, **kwargs):
        """The `RankGenerator` class produces a hyper-rectangle for a given set of
        tensor, pipeline, data, expert, and context parallelism. If we have an encoder,
        in addition to the default decoder, we essentially instantiate two `RankGenerator`
        classes to construct the parallelism for each module separately, and we then have
        to stitch them together for the right groups. For now, this means pp and tp-pp."""
        d_ranks = decoder_rank_generator.get_ranks(group_type, **kwargs)
        if encoder_rank_generator is None:
            for x in d_ranks:
                yield x
            return
        e_ranks = encoder_rank_generator.get_ranks(group_type, **kwargs)
        if group_type == 'pp':
            # Map 1 encoder tp rank to several decoder tp ranks, because
            # these won't be the same size.
            for x, y in zip(cycle(e_ranks), d_ranks):
                yield x + y
        elif group_type == 'tp-pp':
            # For this group, we can just return the concatenated
            # groups together, because their sizes are the same.
            assert len(e_ranks) == len(d_ranks)
            for x, y in zip(e_ranks, d_ranks):
                yield x + y
        else:
            for x in e_ranks:
                yield x
            for x in d_ranks:
                yield x
    
    timeout = timedelta(minutes=distributed_timeout_minutes)

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)


    global _ULYSSES_SP_PARALLEL_GROUP
    global _ULYSSES_SP_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_USP
    global _DATA_PARALLEL_GROUP_WITH_USP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_USP
    global _DATA_PARALLEL_GROUP_WITH_USP_CP
    global _DATA_PARALLEL_GROUP_WITH_USP_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_USP_CP
    for ranks_with_usp_cp in generator_wrapper('dp-usp-cp'):
        group_with_usp_cp = torch.distributed.new_group(
            ranks_with_usp_cp, timeout=timeout, pg_options=PS.get_nccl_options('dp_usp_cp', nccl_comm_cfgs)
        )
        group_with_usp_cp_gloo = torch.distributed.new_group(
            ranks_with_usp_cp, timeout=timeout, backend="gloo"
        )
        if rank in ranks_with_usp_cp:
            _DATA_PARALLEL_GROUP_WITH_USP_CP = group_with_usp_cp
            _DATA_PARALLEL_GROUP_WITH_USP_CP_GLOO = group_with_usp_cp_gloo
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_USP_CP = ranks_with_usp_cp
            

    # Apply SHARP to DP process groups
    if use_sharp:
        if rank == 0:
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
            group=get_data_parallel_group(with_context_parallel=True, with_ulysses_sp_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
        # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
        os.environ["NCCL_COLLNET_ENABLE"] = "0"

    # Build the ulysses-sequence-parallel groups.
    for ranks_with_usp in generator_wrapper('dp-usp'):
        group_with_usp = torch.distributed.new_group(
            ranks_with_usp, timeout=timeout, pg_options=PS.get_nccl_options('dp_usp', nccl_comm_cfgs)
        )
        group_with_usp_gloo = torch.distributed.new_group(
            ranks_with_usp, timeout=timeout, backend="gloo"
        )
        if rank in ranks_with_usp:
            _DATA_PARALLEL_GROUP_WITH_USP = group_with_usp
            _DATA_PARALLEL_GROUP_WITH_USP_GLOO = group_with_usp_gloo
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_USP = ranks_with_usp



    # Build the ulysses-sp-parallel groups.
    global _ULYSSES_SP_PARALLEL_GROUP
    global _ULYSSES_SP_PARALLEL_GLOBAL_RANKS
    assert _ULYSSES_SP_PARALLEL_GROUP is None, 'ulysses parallel group is already initialized'
    for ranks in generator_wrapper('usp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=PS.get_nccl_options('usp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _ULYSSES_SP_PARALLEL_GROUP = group
            _ULYSSES_SP_PARALLEL_GLOBAL_RANKS = ranks
    

def get_data_parallel_group(with_context_parallel=False, with_ulysses_sp_parallel=False):
    """Get the data parallel group the caller rank belongs to."""

    if with_context_parallel and with_ulysses_sp_parallel:
        if _DATA_PARALLEL_GROUP_WITH_USP_CP is None:
            return PS.get_data_parallel_group(with_context_parallel)
        else:
            return _DATA_PARALLEL_GROUP_WITH_USP_CP
    else:
        return PS.get_data_parallel_group(with_context_parallel)


def get_data_parallel_group_gloo(with_context_parallel=False, with_ulysses_sp_parallel=False):
    """Get the data parallel group-gloo the caller rank belongs to."""
    if with_context_parallel and with_ulysses_sp_parallel:
        if _DATA_PARALLEL_GROUP_WITH_USP_CP_GLOO is None:
            return PS.get_data_parallel_group_gloo(with_context_parallel)
        else:
            return _DATA_PARALLEL_GROUP_WITH_USP_CP_GLOO
    else:
        return PS.get_data_parallel_group_gloo(with_context_parallel)


def get_ulysses_sp_parallel_group(check_initialized=False):
    """Get the ulysses sequence parallel group the caller rank belongs to."""
    # TODO: support hetero

    if check_initialized:
        _ULYSSES_SP_PARALLEL_GROUP is not None, 'ulysses-sp parallel group is not initialized'
    return _ULYSSES_SP_PARALLEL_GROUP


def get_ulysses_sp_parallel_global_ranks(check_initialized=False):
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    # TODO: support hetero

    if check_initialized:
        assert (
            _ULYSSES_SP_PARALLEL_GLOBAL_RANKS is not None
        ), 'ulysses sequence parallel group is not initialized'
    return _ULYSSES_SP_PARALLEL_GLOBAL_RANKS


def get_data_parallel_src_rank(with_context_parallel=False, with_ulysses_sp_parallel=False):
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    
    if with_context_parallel and with_ulysses_sp_parallel:
        if _DATA_PARALLEL_GLOBAL_RANKS_WITH_USP_CP is None:
            return PS.get_data_parallel_src_rank(with_context_parallel)
        else:
            return _DATA_PARALLEL_GLOBAL_RANKS_WITH_USP_CP[0]
    else:
        return PS.get_data_parallel_src_rank(with_context_parallel)


def get_data_parallel_rank(with_context_parallel=False, with_ulysses_sp_parallel=False):
    """Return my rank for the data parallel group."""
    if with_context_parallel and with_ulysses_sp_parallel:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(
                group=get_data_parallel_group(with_context_parallel=with_context_parallel, with_ulysses_sp_parallel=with_ulysses_sp_parallel)
            )
        else:
            return 0
    else:
        return PS.get_data_parallel_rank(with_context_parallel)


def get_ulysses_sp_parallel_world_size():
    """Return world size for the ulysses_sp parallel group."""
    # TODO: hete case

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if get_ulysses_sp_parallel_group() is None:
            return 1
        return torch.distributed.get_world_size(group=get_ulysses_sp_parallel_group())
    else:
        return 0


def get_ulysses_sp_parallel_rank():
    """Return my rank for the ulysses_sp parallel group."""
    # TODO: hete case
    
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_ulysses_sp_parallel_group())
    else:
        return 0


def destroy_model_parallel():
    """Set the groups to none."""
    PS.destroy_model_parallel()
    global _ULYSSES_SP_PARALLEL_GROUP
    _ULYSSES_SP_PARALLEL_GROUP = None
    global _ULYSSES_SP_PARALLEL_GLOBAL_RANKS
    _ULYSSES_SP_PARALLEL_GLOBAL_RANKS = None
    global _DATA_PARALLEL_GROUP_WITH_USP
    _DATA_PARALLEL_GROUP_WITH_USP = None
    global _DATA_PARALLEL_GROUP_WITH_USP_GLOO
    _DATA_PARALLEL_GROUP_WITH_USP_GLOO = None
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_USP
    _DATA_PARALLEL_GLOBAL_RANKS_WITH_USP = None
    global _DATA_PARALLEL_GROUP_WITH_USP_CP
    _DATA_PARALLEL_GROUP_WITH_USP_CP = None
    global _DATA_PARALLEL_GROUP_WITH_USP_CP_GLOO
    _DATA_PARALLEL_GROUP_WITH_USP_CP_GLOO = None
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_USP_CP
    _DATA_PARALLEL_GLOBAL_RANKS_WITH_USP_CP = None
