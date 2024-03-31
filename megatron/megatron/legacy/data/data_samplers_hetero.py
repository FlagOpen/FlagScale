# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# This file is modified to support heterogenous training

"""Dataloaders."""


import random
import torch
import numpy as np
from torch.utils.data import Dataset
from megatron.training import get_args
from megatron.training import get_hetero_context
from megatron.core import mpu


def build_pretraining_data_loader_hetero(dataset, consumed_samples):
    """Buld dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()
    hetero_context = get_hetero_context()

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            hetero_micro_batch_sizes=args.hetero_micro_batch_sizes,
            hetero_data_parallel_splits=args.hetero_data_parallel_splits,
            hetero_context=hetero_context,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            hetero_micro_batch_sizes=args.hetero_micro_batch_sizes,
            hetero_data_parallel_splits=args.hetero_data_parallel_splits,
            hetero_context=hetero_context,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding)
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True)

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 hetero_micro_batch_sizes, hetero_data_parallel_splits, hetero_context,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.hetero_micro_batch_sizes = hetero_micro_batch_sizes
        self.hetero_data_parallel_splits = hetero_data_parallel_splits
        self.hetero_context = hetero_context
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_for_all_data_parallel = sum(map(lambda x, y: x * y, 
                                                     hetero_micro_batch_sizes,
                                                     hetero_data_parallel_splits))
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert data_parallel_size == sum(self.hetero_data_parallel_splits)
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        accumulated_mbs = 0
        accumulated_ranks = 0
        current_micro_batch_size = 0
        data_parallel_rank = self.data_parallel_rank 
        for mbs, split in zip(self.hetero_micro_batch_sizes,
                              self.hetero_data_parallel_splits):
            current_micro_batch_size = mbs 
            if data_parallel_rank < accumulated_ranks + split:
                break
            else:
                accumulated_mbs += mbs * split
                accumulated_ranks += split
        start_idx = accumulated_mbs + (data_parallel_rank - accumulated_ranks) * current_micro_batch_size 
        end_idx = start_idx + current_micro_batch_size

        # TODO: need to be removed after debugging
        # print(f'physical_rank: {torch.distributed.get_rank()}, '\
        #       f'logical_rank: {self.hetero_context.to_logical_ranks([torch.distributed.get_rank()])[0]}, '\
        #       f'logical_dp_rank: {self.ata_parallel_rank}, '\
        #       f'start_idx: {start_idx}, end_idx: {end_idx}, '\
        #       f'cur_mbs: {current_micro_batch_size}, mbs: {self.micro_batch_size}',
        #       flush=True)

        assert current_micro_batch_size == self.micro_batch_size, \
            'current micro batch size ({}) is not equal to micro batch size ({})'.format(
                current_micro_batch_size, self.micro_batch_size)

        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_for_all_data_parallel:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class RandomSeedDataset(Dataset):

    def __init__(self, dataset):
        args = get_args()
        self.base_seed = args.seed
        self.curr_seed = args.seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 hetero_micro_batch_sizes, hetero_data_parallel_splits, hetero_context,
                 data_parallel_rank, data_parallel_size, data_sharding):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.hetero_micro_batch_sizes = hetero_micro_batch_sizes
        self.hetero_data_parallel_splits = hetero_data_parallel_splits
        self.hetero_context = hetero_context
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_for_all_data_parallel = sum(map(lambda x, y: x * y, 
                                                     hetero_micro_batch_sizes,
                                                     hetero_data_parallel_splits))
        self.last_batch_size = \
            self.total_samples % self.micro_batch_for_all_data_parallel

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert data_parallel_size == sum(self.hetero_data_parallel_splits)
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_for_all_data_parallel == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            micro_steps =  self.total_samples // self.micro_batch_for_all_data_parallel
            bucket_size = micro_steps * self.micro_batch_size
            current_micro_steps = current_epoch_samples // self.micro_batch_for_all_data_parallel 
            bucket_offset = current_micro_steps * self.micro_batch_size 

            accumulated_samples = 0
            accumulated_ranks = 0
            current_micro_batch_size = 0
            data_parallel_rank = self.hetero_context.to_logical_ranks([self.data_parallel_rank])[0]
            for  mbs, split in zip(self.hetero_micro_batch_sizes,
                                   self.hetero_data_parallel_splits):
                current_micro_batch_size = mbs 
                if data_parallel_rank < accumulated_ranks + split:
                    break
                else:
                    accumulated_samples += mbs * split * micro_steps 
                    accumulated_ranks += split
            
            assert current_micro_batch_size == self.micro_batch_size, \
                'current micro batch size ({}) is not equal to micro batch size ({})'.format(
                    current_micro_batch_size, self.micro_batch_size)
            start_idx = accumulated_samples + (data_parallel_rank - accumulated_ranks) * current_micro_batch_size * micro_steps

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            micro_steps =  self.total_samples // self.micro_batch_for_all_data_parallel
            full_bucket_size = micro_steps * self.micro_batch_for_all_data_parallel
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = \
                torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = []
            idx_nums = [ mbs * micro_steps for mbs in self.hetero_micro_batch_sizes]
            for i, idx in enumerate(idx_range_active):
                finished = sum (1 for num in idx_nums if num == 0)
                unfinished = sum (1 for num in idx_nums if num > 0)
                rank = finished + i % unfinished 
                data_parallel_rank = self.hetero_context.to_logical_ranks([self.data_parallel_rank])[0]
                if idx_nums[rank] != 0:
                    if rank == data_parallel_rank:
                        idx_range.append(idx)
                    idx_nums[rank] -= 1

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_for_all_data_parallel
                yield batch
                batch = []
