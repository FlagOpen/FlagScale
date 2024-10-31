# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Dict, Generator, Generic, Iterator, List, Optional, Tuple, TypeVar, Union

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import (
    BaseSingleWrapperDataset,
    BaseSingleWrapperMergedState,
    BaseSingleWrapperState,
    get_sample_restore_key,
)

T_sample = TypeVar("T_sample")


@dataclass
class SampleBufferState(BaseSingleWrapperState):
    buffer: List[Tuple[Union[str, int], ...]]


@dataclass
class SampleBufferMergedState(BaseSingleWrapperMergedState):
    buffer: List[List[Tuple[Union[str, int], ...]]]


class SavableSampleBuffer(BaseSingleWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """A buffer of samples, savable."""

    _buffer: List[List[T_sample]]
    _restore_keys: List[List[Tuple[Union[str, int, tuple], ...]]]

    _restore_pending: bool = False

    worker_config: WorkerConfig
    __rank_id: Optional[int] = None

    def __init__(self, dataset: SavableDataset[T_sample], worker_config: WorkerConfig):
        super().__init__(dataset)
        self.worker_config = worker_config
        self._buffer = [[] for _ in range(max(worker_config.num_workers, 1))]
        self._restore_keys = [[] for _ in range(max(worker_config.num_workers, 1))]

    @property
    def _rank_id(self) -> int:
        if self.__rank_id is None:
            self.worker_config.assert_worker()
            self.__rank_id = self.worker_config.rank_worker_id()
        return self.__rank_id

    def worker_start(self) -> None:
        # Cleanup other states
        for i in range(self.worker_config.num_workers):
            if i != self._rank_id:
                self._buffer[i].clear()
                self._restore_keys[i].clear()
        if self._restore_pending:
            assert len(self._buffer[self._rank_id]) == 0
            self._restore_pending = False
            for restore_key in self._restore_keys[self._rank_id]:
                self._buffer[self._rank_id].append(self.restore_sample(restore_key))
        assert len(self._buffer[self._rank_id]) == len(self._restore_keys[self._rank_id])

    def append(self, sample: T_sample) -> T_sample:
        self._buffer[self._rank_id].append(sample)
        self._restore_keys[self._rank_id].append(get_sample_restore_key(sample))
        return sample

    def extend(self, samples: List[T_sample]) -> None:
        self._buffer[self._rank_id].extend(samples)
        self._restore_keys[self._rank_id].extend(
            get_sample_restore_key(sample) for sample in samples
        )

    def append_iter(self) -> Generator[T_sample, None, None]:
        for sample in self.dataset:
            yield self.append(sample)

    def pop(self, index: int) -> T_sample:
        self._restore_keys[self._rank_id].pop(index)
        return self._buffer[self._rank_id].pop(index)

    def __iter__(self) -> Iterator[T_sample]:
        return iter(self._buffer[self._rank_id])

    def __getitem__(self, index: Union[int, slice]) -> T_sample:
        return self._buffer[self._rank_id][index]

    def __setitem__(self, index: Union[int, slice], value: T_sample) -> None:
        self._buffer[self._rank_id][index] = value
        if isinstance(index, slice):
            self._restore_keys[self._rank_id][index] = (get_sample_restore_key(v) for v in value)
        else:
            self._restore_keys[self._rank_id][index] = get_sample_restore_key(value)

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self._buffer[self._rank_id][index]
        del self._restore_keys[self._rank_id][index]

    def __len__(self) -> int:
        return len(self._restore_keys[self._rank_id])

    def save_state(self) -> SampleBufferState:
        self.dataset.assert_can_restore()
        return SampleBufferState.extend(
            super().save_state(),
            buffer=list(self._restore_keys[self._rank_id]),
        )

    def merge_states(self, states: List[Optional[SampleBufferState]]) -> SampleBufferMergedState:
        assert all(s is None or isinstance(s, SampleBufferState) for s in states)
        return SampleBufferMergedState.extend(
            super().merge_states(states),
            buffer=[[] if s is None else s.buffer for s in states],
        )

    def restore_state(self, state: Optional[SampleBufferMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._buffer = [[] for _ in range(max(self.worker_config.num_workers, 1))]
            self._restore_keys = [[] for _ in range(max(self.worker_config.num_workers, 1))]
            self._restore_pending = False
        else:
            assert isinstance(state, SampleBufferMergedState)
            assert len(state.buffer) == max(self.worker_config.num_workers, 1)
            self._buffer = [[] for _ in range(max(self.worker_config.num_workers, 1))]
            self._restore_keys = state.buffer
            self._restore_pending = True

    def restore_key(self) -> Tuple[Union[str, int], ...]:
        return tuple(self._restore_keys[self._rank_id])

    def restore_samples(
        self, index: Tuple[Union[str, int, tuple], ...]
    ) -> Tuple[Tuple[Union[str, int, tuple], ...], List[T_sample]]:
        buffer = []
        restore_keys = []
        for sub_index in index:
            sample = self.restore_sample(sub_index)
            restore_keys.append(get_sample_restore_key(sample))
            buffer.append(sample)
        return tuple(restore_keys), buffer

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        return self.dataset.restore_sample(index)

    def clear(self) -> None:
        self._buffer[self._rank_id].clear()
        self._restore_keys[self._rank_id].clear()

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "worker_config": self.worker_config.config(),
        }

    def __str__(self):
        return f"SavableSampleBuffer(size={len(self._buffer)})"
