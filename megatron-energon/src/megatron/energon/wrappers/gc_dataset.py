# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import gc
from typing import Any, Dict, Generic, Iterator, TypeVar

import torch
import torch.utils.data
import torch.utils.data.dataloader
from torch.distributed._shard.sharded_tensor import ShardedTensorBase
from torch.distributed.distributed_c10d import reduce_op

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.wrappers.base import BaseSingleWrapperDataset

T_sample = TypeVar("T_sample")

_frozen_cuda_tensors = set()
_frozen_cuda_tensors_initialized = False


class GcFreezeError(RuntimeError):
    pass


def gc_init_worker(worker_id: int):
    """This function should be called by any forked worker process that uses CUDA.
    It should be called as early as possible in the worker process, ideally in
    the worker_init_fn of the DataLoader.

    By keeping a reference to all CUDA tensors in the worker process, we can
    prevent the forked tensors from being garbage collected."""

    global _frozen_cuda_tensors_initialized, _frozen_cuda_tensors

    num_tensors = 0
    for o in gc.get_objects():
        try:
            if o is not reduce_op:
                if isinstance(o, torch.Tensor):
                    if isinstance(o, ShardedTensorBase) or o.is_cuda:
                        # Calling .is_cuda or any hasattr on ShardedTensor will raise an error
                        # Hence, o.is_cuda is only called if o is not a ShardedTensor (in the if above)

                        _frozen_cuda_tensors.add(o)
                        num_tensors += 1
                elif isinstance(o, torch.utils.data.dataloader._MultiProcessingDataLoaderIter):
                    o._shutdown = True
        except ReferenceError:
            # Can happen if the object is a weakref proxy, don't care
            pass

    _frozen_cuda_tensors_initialized = True


class GcDataset(BaseSingleWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """Applies a garbage collection step. This is needed, because python garbage collection
    does not work well with very large objects, such as tensors. This case happens, if there are
    a few hundred objects created and released every epoch (some of them being (large) tensors),
    where a lot of them are alive at the same time, but released later. In that case, those objects
    may end up in gc generation 2, where they may live until a lot of objects have been created,
    until automatic garbage collection of gen2 is actually triggered. To avoid this memory leak,
    `gc.collect()` is best to be called regularly. In addition, if `gc.freeze()` is used before the
    loop, it will remove the objects currently alive from garbage collection checks, thus making the
    gc faster.
    """

    every_n_iter: int
    freeze: bool

    def __init__(
        self, dataset: SavableDataset[T_sample], every_n_iter: int = 1, freeze: bool = False
    ):
        """Construct a GcDataset, which applies garbage collection every `every_n_iter` iterations.

        Args:
            dataset: The input dataset to wrap
            every_n_iter: How often to perform garbage collection
            freeze: If true, run `gc.freeze()` before the loop, and `gc.unfreeze()` after the loop.
                This will speed up garbage collection, but will keep all initially alive objects
                alive until the end of the loop (i.e. if the dataset state was restored, that state
                will be saved as well).
        """
        super().__init__(dataset)
        self.every_n_iter = every_n_iter
        self.freeze = freeze

    def __len__(self):
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_sample]:
        in_worker = torch.utils.data.get_worker_info() is not None
        if in_worker and not _frozen_cuda_tensors_initialized:
            raise GcFreezeError(
                "You are using GcDataset with multiple workers, but forgot to call gc_init_worker() in at least one forked worker process."
            )

        if self.freeze:
            gc.collect()
            gc.freeze()
        try:
            iter = 0
            for sample in self.dataset:
                yield sample
                iter += 1
                if iter >= self.every_n_iter:
                    gc.collect()
                    iter = 0
        finally:
            if self.freeze:
                gc.unfreeze()

    def config(self) -> Dict[str, Any]:
        # This is transparent, no config to be saved (it does not affect the dataset)
        return self.dataset.config()

    def __str__(self):
        return f"GcDataset(every_n_iter={self.every_n_iter}, dataset={self.dataset})"
