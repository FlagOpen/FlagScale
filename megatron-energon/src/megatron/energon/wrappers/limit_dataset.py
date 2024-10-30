# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import (
    BaseSingleWrapperDataset,
    BaseSingleWrapperMergedState,
    BaseSingleWrapperState,
)

T_sample = TypeVar("T_sample")


@dataclass
class LimitState(BaseSingleWrapperState):
    offset: int


@dataclass
class LimitMergedState(BaseSingleWrapperMergedState):
    offset: List[int]


class LimitDataset(BaseSingleWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """Limits the length of the dataset."""

    dataset: SavableDataset[T_sample]
    length: int
    worker_config: WorkerConfig

    _current_offset: List[int]

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        length: int,
        *,
        reset_after_epoch: bool = False,
        worker_config: WorkerConfig,
    ):
        """
        Limits the length of the dataset.

        Args:
            dataset: The dataset to limit
            length: The length to limit to
            reset_after_epoch: If true, reset the underlying dataset after one epoch.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset)
        self.length = length
        self.reset_after_epoch = reset_after_epoch
        self.worker_config = worker_config
        self._current_offset = [0] * max(self.worker_config.num_workers, 1)

    def __len__(self) -> int:
        return min(self.length, len(self.dataset))

    def __iter__(self) -> Iterator[T_sample]:
        worker_id = self.worker_config.rank_worker_id()

        # Compute the local limit for this worker, i.e. all worker's limits sum up to the total
        if self.worker_config.num_workers <= 1:
            local_limit = self.length
        else:
            local_limit = self.length // self.worker_config.num_workers
            if worker_id < self.length % self.worker_config.num_workers:
                local_limit += 1

        if self.worker_config.should_log(level=2):
            self.worker_config.worker_log(
                {
                    "t": "LimitDataset.start",
                    "r": self.worker_config.rank,
                    "w": worker_id,
                    "offset": self._current_offset[worker_id],
                    "local_limit": local_limit,
                    "limit": self.length,
                }
            )

        offset_range = list(
            range(self._current_offset[self.worker_config.rank_worker_id()], local_limit)
        )
        # Only iterate self.dataset if there are samples to iterate
        if len(offset_range) > 0:
            for sample, offset in zip(
                self.dataset,
                offset_range,
            ):
                self._current_offset[worker_id] = offset + 1
                yield sample

        if self.worker_config.should_log(level=2):
            self.worker_config.worker_log(
                {
                    "t": "LimitDataset.done",
                    "r": self.worker_config.rank,
                    "w": worker_id,
                    "offset": self._current_offset[worker_id],
                    "local_limit": local_limit,
                    "limit": self.length,
                }
            )

        # Reset the inner dataset
        self.dataset.restore_state(None)
        self._current_offset = [0] * max(self.worker_config.num_workers, 1)
        if self.reset_after_epoch:
            self.dataset.restore_state(None)

    def worker_has_samples(self) -> bool:
        return super().worker_has_samples() and self.length > 0

    def save_state(self) -> LimitState:
        return LimitState.extend(
            super().save_state(),
            offset=self._current_offset[self.worker_config.rank_worker_id()],
        )

    def merge_states(self, states: List[LimitState]) -> LimitMergedState:
        assert all(s is None or isinstance(s, LimitState) for s in states)
        return LimitMergedState.extend(
            super().merge_states(states),
            offset=[0 if s is None else s.offset for s in states],
        )

    def restore_state(self, state: Optional[LimitMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._current_offset = [0] * max(self.worker_config.num_workers, 1)
        else:
            assert isinstance(state, LimitMergedState)
            self._current_offset = state.offset

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "length": self.length,
            "reset_after_epoch": self.reset_after_epoch,
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"LimitDataset(length={self.length}, dataset={self.dataset})"
