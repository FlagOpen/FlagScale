# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import json
import multiprocessing
import random
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, TextIO, TypeVar

import numpy
import torch.distributed
import torch.utils.data

from megatron.energon.epathlib import EPath

__all__ = ("WorkerConfig",)

T = TypeVar("T")


@dataclass(eq=False)
class WorkerConfig:
    """
    Provides information about the current worker and the global configuration. This gives each
    data parallel rank its proper config. Every `rank` (up to `world_size-1`) must be used.
    If set wrong, the datasets might yield the same data or data might be missing, as data
    is split over the data parallel ranks with this config!
    You may set the same rank, if you need multiple ranks to retrieve the same data.
    """

    #: The data parallel rank/id of the current process.
    rank: int
    #: The total number of data parallel processes.
    world_size: int
    #: The number of workers per rank. May be 0 to disable worker processes.
    num_workers: int

    #: If not using all ranks for data parallel, set this to the corresponding group.
    data_parallel_group: Optional[torch.distributed.ProcessGroup] = None

    #: The id offset of the current worker. e.g. the worker may live as `worker_info.id=0`, but
    # actually yield samples for id=1 (i.e. worker_id_offset=1). Required to support restoring the
    # worker state if last emitted sample was not for worker_id=0. Required by SavableDataLoader to
    # restore the worker state. Is only set to nonzero within a worker process.
    worker_id_offset: ClassVar[int] = 0

    #: The following seed_offset is used used at two points in the code.
    # 1. The seed_offset in the worker_config that is passed to the dataset initialization, is used
    #    to set the seed for the dataset shuffling and shuffled blending (All code that uses WorkerRng).
    # 2. The worker_config passed to the data loader initialization, is used to set the seed for the
    #    torch, numpy and random libraries. This does not affect the dataset shuffling, but only the
    #    user code (e.g. code in TaskEncoder).
    # Typically, you will not have to worry about this, and just use the same worker_config for both.
    # However, if you need to, you can pass two different worker_configs to the dataset and the
    # data loader.
    seed_offset: int = 0

    #: The path to the debug file for the current worker. Should contain "{worker_id}" and "{pid}"
    # to separate the workers.
    worker_debug_path: Optional[str] = None
    #: Log level for worker logging.
    worker_log_level: int = 0
    #: The opened file for the current worker. Should not be set from outside.
    _worker_debug_file: Optional[TextIO] = None
    #: worker_id of the opened worker debug file
    _worker_debug_file_worker_id: Optional[int] = None

    #: The current sample index within the current iterating worker
    _sample_index_stack: ClassVar[Optional[List[int]]] = None
    #: The current worker config within the current iterating worker
    active_worker_config: ClassVar[Optional["WorkerConfig"]] = None

    #: The global rank override for the worker. Required for restoring samples.
    _worker_override_global_rank: ClassVar[Optional[List[int]]] = None

    def worker_activate(self, sample_index: int, override_global_rank: Optional[int] = None):
        """Activates the worker config for the current worker and sets it as actively iterating.
        Must be called before next() call on the datasets."""
        assert WorkerConfig.active_worker_config is None
        WorkerConfig._sample_index_stack = [sample_index]
        WorkerConfig.active_worker_config = self
        WorkerConfig._worker_override_global_rank = override_global_rank

    def worker_push_sample_index(self, sample_index: int):
        """Pushes a new sample index to the sample index stack. Should be set by wrapping datasets
        before calling inners."""
        assert WorkerConfig.active_worker_config is not None
        WorkerConfig._sample_index_stack.append(sample_index)

    def worker_pop_sample_index(self):
        """Pushes a new sample index to the sample index stack. Should be set by wrapping datasets
        before calling inners."""
        assert WorkerConfig.active_worker_config is not None
        return WorkerConfig._sample_index_stack.pop()

    def worker_deactivate(self):
        """Deactivates the worker config for the current worker and deactivates it for iterating.
        Must be called after next() call on the datasets."""
        if WorkerConfig.active_worker_config is not None:
            assert (
                len(WorkerConfig._sample_index_stack) == 1
            ), f"Sample index stack not empty: {WorkerConfig._sample_index_stack}"
            WorkerConfig._sample_index_stack = None
            WorkerConfig.active_worker_config = None
            WorkerConfig._worker_override_global_rank = None

    @property
    def active_worker_sample_index(self) -> int:
        """Returns the current sample index for the actively iterating worker."""
        # Internal sample index is for the local worker. If using multiple workers per rank, this
        # must be multiplied by the number of workers and offset by the local worker index.
        return (
            WorkerConfig._sample_index_stack[-1] * max(self.num_workers, 1) + self.rank_worker_id()
        )

    @property
    def active_worker_batch_index(self) -> int:
        """Returns the current batch index for the actively iterating worker."""
        # Internal batch index is for the local worker. If using multiple workers per rank, this
        # must be multiplied by the number of workers and offset by the local worker index.
        return (
            WorkerConfig._sample_index_stack[0] * max(self.num_workers, 1) + self.rank_worker_id()
        )

    def __eq__(self, other):
        """Do not compare everything to check for equal config"""
        if not isinstance(other, WorkerConfig):
            return NotImplementedError()
        return all(
            [
                self.rank == other.rank,
                self.world_size == other.world_size,
                self.num_workers == other.num_workers,
            ]
        )

    @staticmethod
    def default_worker_config(
        num_workers: int = 4, data_parallel_group: Optional[torch.distributed.ProcessGroup] = None
    ) -> "WorkerConfig":
        """Returns the default worker config using torch distributed."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank(data_parallel_group)
            world_size = torch.distributed.get_world_size(data_parallel_group)
        else:
            rank = 0
            world_size = 1
        return WorkerConfig(
            rank=rank,
            world_size=world_size,
            num_workers=num_workers,
            data_parallel_group=data_parallel_group,
        )

    def rank_worker_id(self) -> int:
        """Returns the self worker id within the current rank."""
        if self._worker_override_global_rank:
            assert self.worker_id_offset == 0
            return self._worker_override_global_rank % self.num_workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.worker_id_offset
        assert worker_info.num_workers == self.num_workers
        return (
            worker_info.id + worker_info.num_workers - self.worker_id_offset
        ) % worker_info.num_workers

    def assert_worker(self):
        """Checks if the current process is a worker (if configured so), and that the workers are
        properly configured."""
        if self.num_workers <= 1:
            assert self.rank_worker_id() == 0
        else:
            worker_info = torch.utils.data.get_worker_info()
            assert worker_info is not None, "Cannot iterate out of worker context"
            assert worker_info.num_workers == self.num_workers, (
                f"Actual number of workers for this rank ({worker_info.num_workers}) does not "
                f"match the configured number of workers ({self.num_workers})"
            )

    def global_worker_id(self, override_local_worker_id: Optional[int] = None) -> int:
        """Returns the global worker index by multiplying the rank with the number of workers.
        Alternatively, you can override the local worker id.

        Args:
            override_local_worker_id (int, optional): The local worker id to override. None means
                the current worker, which is the default.
        """
        if self._worker_override_global_rank is not None:
            assert override_local_worker_id is None
            return self._worker_override_global_rank

        if override_local_worker_id is not None:
            return self.rank * self.num_workers + override_local_worker_id
        else:
            self.assert_worker()
            return self.rank * self.num_workers + self.rank_worker_id()

    def worker_seed(self, override_local_worker_id: Optional[int] = None) -> int:
        """Returns the seed for the current worker (or a specified worker).
        Base on the current worker id and the seed offset, compute a seed.
        Alternatively, you can override the local worker id with a fixed one to
        pregenerate seeds for multiple workers.

        Args:
            override_local_worker_id (int, optional): The local worker id to override. None means
                the current worker, which is the default.
        """

        if self.num_workers == 0:
            # If we are not using workers, different ranks should still get a different seed
            global_worker_id = self.rank
        else:
            global_worker_id = self.global_worker_id(override_local_worker_id)

        seed_offset = self.seed_offset

        seed_hash = hashlib.sha1(f"{global_worker_id},{seed_offset}".encode("utf-8")).digest()

        return int.from_bytes(seed_hash, byteorder="big", signed=False) & 0xFFFFFFFF

    def config(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "num_workers": self.num_workers,
            "data_parallel_group": (
                self.data_parallel_group.size() if self.data_parallel_group else None
            ),
        }

    def should_log(self, level: int) -> bool:
        return level <= self.worker_log_level

    def worker_log(self, data: dict) -> None:
        """Logs the given data to the worker debug file."""
        if self.worker_debug_path is None:
            print(json.dumps(data) + "\n", end="", flush=True)
        else:
            in_worker = torch.utils.data.get_worker_info() is not None
            # Additional "worker" with rank_worker_id=0 is the main process. All workers have +1
            # as their worker_id.
            worker_id = (
                self.rank * (self.num_workers + 1) + self.rank_worker_id() + (1 if in_worker else 0)
            )
            if self._worker_debug_file is None or self._worker_debug_file_worker_id != worker_id:
                if self._worker_debug_file is not None:
                    self._worker_debug_file.close()
                path = EPath(
                    self.worker_debug_path.format(
                        worker_id=worker_id, pid=multiprocessing.current_process().ident
                    )
                )
                path.parent.mkdir(exist_ok=True, parents=True)
                self._worker_debug_file = path.open("w")
                self._worker_debug_file_worker_id = worker_id
            self._worker_debug_file.write(json.dumps(data) + "\n")
            self._worker_debug_file.flush()
