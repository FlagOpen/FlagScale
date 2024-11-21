# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Dict, Generator, Generic, Iterator, List, Optional, TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import (
    BaseSingleWrapperDataset,
    BaseSingleWrapperMergedState,
    BaseSingleWrapperState,
)

T_sample = TypeVar("T_sample")


@dataclass
class EpochizeState(BaseSingleWrapperState):
    offset: int


@dataclass
class EpochizeMergedState(BaseSingleWrapperMergedState):
    offset: List[int]


class EpochizeDataset(BaseSingleWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """
    Uses the base dataset, and creates one epoch, which has length samples. Keeps the underlying
    dataset iterator alive over epochs (i.e. if it is an infinite dataset, it will keep the state).
    Repeats the underlying dataset if the iterator is exhausted.
    """

    length: int
    worker_config: WorkerConfig
    _active_iter: Optional[Iterator[T_sample]]

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        length: int,
        worker_config: WorkerConfig,
    ):
        """
        Create the epochized dataset.

        Args:
            dataset: The source dataset (possibly infinite)
            length: Number of samples to iterate before iteration stops (i.e. one epoch). When
                iteration continues, the original dataset iterator is resumed and does only restart
                if exhausted.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset)
        self.length = length
        self.worker_config = worker_config
        self._offset = [0] * max(self.worker_config.num_workers, 1)
        self._active_iter = None

    def _infinite(self) -> Generator[T_sample, None, None]:
        while True:
            for sample in self.dataset:
                yield sample

    def __iter__(self) -> Iterator[T_sample]:
        # Compute the local length for this worker, i.e. all worker's lengths sum up to the total
        worker_idx = self.worker_config.rank_worker_id()

        if self.worker_config.num_workers <= 1:
            local_length = self.length
        else:
            local_length = self.length // self.worker_config.num_workers
            if self.worker_config.rank_worker_id() < self.length % self.worker_config.num_workers:
                local_length += 1

        if self.worker_config.should_log(level=2):
            self.worker_config.worker_log(
                {
                    "t": "EpochizeDataset.epoch_start",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "offset": self._offset[worker_idx],
                    "local_length": local_length,
                    "length": self.length,
                }
            )

        offset_range = list(range(self._offset[worker_idx], local_length))

        # Only iterate if there are samples to iterate
        if len(offset_range) > 0:
            if self._active_iter is None:
                self._active_iter = iter(self._infinite())

            for idx in offset_range:
                self._offset[worker_idx] = (idx + 1) % local_length
                yield next(self._active_iter)

        if self.worker_config.should_log(level=2):
            self.worker_config.worker_log(
                {
                    "t": "EpochizeDataset.epoch_end",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "offset": self._offset[worker_idx],
                    "local_length": local_length,
                    "length": self.length,
                }
            )

    def __len__(self) -> int:
        return self.length

    def save_state(self) -> EpochizeState:
        return EpochizeState.extend(
            super().save_state(), offset=self._offset[self.worker_config.rank_worker_id()]
        )

    def merge_states(self, states: List[EpochizeState]) -> EpochizeMergedState:
        assert all(s is None or isinstance(s, EpochizeState) for s in states)
        return EpochizeMergedState.extend(
            super().merge_states(states),
            offset=[0 if state is None else state.offset for state in states],
        )

    def restore_state(self, state: Optional[EpochizeMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._offset = [0] * max(self.worker_config.num_workers, 1)
        else:
            assert isinstance(state, EpochizeMergedState)
            self._offset = state.offset

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "length": self.length,
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"EpochizeDataset(length={self.length}, dataset={self.dataset})"
