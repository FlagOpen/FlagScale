# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from megatron.energon.flavors.base_dataset import MergedState, Sample, SavableDataset, State
from megatron.energon.worker import WorkerConfig

T = TypeVar("T")
T_sample = TypeVar("T_sample")
T_sample_out = TypeVar("T_sample_out")
T_sample_in = TypeVar("T_sample_in")


@dataclass
class BaseSingleWrapperState(State):
    """Base class for dataset states."""

    #: The class name of the dataset saving this state for assertion
    dataset_type: str

    #: State of the inner dataset
    dataset_state: State


@dataclass
class BaseSingleWrapperMergedState(MergedState):
    """Base class for dataset states."""

    #: The class name of the dataset saving/merging this state for assertion
    dataset_type: str

    #: State of the inner dataset
    dataset_state: MergedState


class BaseWrapperDataset(SavableDataset[T_sample], Generic[T_sample]):
    """Base class for dataset wrappers. All dataset wrappers should derive from this. A dataset
    wrapper takes one dataset and modifies its samples to make a new dataset. This can be for
    shuffling samples or applying custom functions to the data. Some wrappers only modify the
    length of the dataset or how it's repeated."""


class BaseSingleWrapperDataset(
    BaseWrapperDataset[T_sample_out], Generic[T_sample_in, T_sample_out]
):
    """Base class for dataset wrappers that wrap a single dataset. Provides default implementations
    for saving and restoring the dataset state."""

    dataset: SavableDataset[T_sample_in]

    def __init__(self, dataset: SavableDataset[T_sample_in]):
        super().__init__()
        self.dataset = dataset

    def save_state(self) -> BaseSingleWrapperState:
        return BaseSingleWrapperState(
            dataset_type=type(self).__name__,
            dataset_state=self.dataset.save_state(),
        )

    def merge_states(
        self, states: Sequence[Optional[BaseSingleWrapperState]]
    ) -> BaseSingleWrapperMergedState:
        assert all(s is None or isinstance(s, BaseSingleWrapperState) for s in states)
        assert all(s is None or s.dataset_type == type(self).__name__ for s in states)
        return BaseSingleWrapperMergedState(
            dataset_type=type(self).__name__,
            dataset_state=self.dataset.merge_states(
                [None if s is None else s.dataset_state for s in states]
            ),
        )

    def restore_state(self, state: Optional[BaseSingleWrapperMergedState]) -> None:
        if state is None:
            self.dataset.restore_state(None)
        else:
            assert isinstance(state, BaseSingleWrapperMergedState)
            assert state.dataset_type == type(self).__name__
            self.dataset.restore_state(state.dataset_state)

    def can_restore_sample(self) -> bool:
        return self.dataset.can_restore_sample()
    
    def assert_can_restore(self) -> None:
        self.dataset.assert_can_restore()

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample_out:
        return self.dataset.restore_sample(index)

    def worker_has_samples(self) -> bool:
        return self.dataset.worker_has_samples()

    def verify_worker_config(self, worker_config: WorkerConfig) -> None:
        super().verify_worker_config(worker_config)
        self.dataset.verify_worker_config(worker_config)


class SampleIndex:
    """A simple class to hold the sample index for each worker."""

    worker_config: WorkerConfig
    _sample_index: List[int]
    __rank: Optional[int] = None

    actives = 0

    def __init__(self, worker_config: WorkerConfig, *, src: Any) -> None:
        self.worker_config = worker_config
        self._sample_index = [0] * max(self.worker_config.num_workers, 1)
        self.src = src

    @property
    def _rank(self) -> int:
        if self.__rank is None:
            self.__rank = self.worker_config.rank_worker_id()
        return self.__rank

    def get_next(self) -> int:
        res = self._sample_index[self._rank]
        self._sample_index[self._rank] += 1
        return res

    @property
    def current_idx(self) -> int:
        return self._sample_index[self._rank]

    @contextmanager
    def ctx(self, sample_idx: Optional[int] = None):
        if sample_idx is None:
            sample_idx = self.get_next()
        assert WorkerConfig.active_worker_config is not None
        WorkerConfig.active_worker_config.worker_push_sample_index(sample_idx)
        # print("  " * SampleIndex.actives + f"Activated from {type(self.src).__name__}({id(self.src)}) {sample_idx} -> {WorkerConfig.active_worker_config._sample_index_stack}")
        SampleIndex.actives += 1
        try:
            yield sample_idx
        finally:
            assert WorkerConfig.active_worker_config is not None
            popped = WorkerConfig.active_worker_config.worker_pop_sample_index()
            SampleIndex.actives -= 1
            # print("  " * SampleIndex.actives + f"Deactivate from {type(self.src).__name__}({id(self.src)}) {sample_idx} -> {WorkerConfig.active_worker_config._sample_index_stack}")
            assert popped == sample_idx, f"Expected {sample_idx}, got {popped}"

    def iter_ctx(
        self,
        it: Iterable[T_sample],
        sample_idx: Optional[int] = None,
    ) -> Generator[Tuple[int, T_sample], None, None]:
        it = iter(it)
        try:
            while True:
                try:
                    with self.ctx(sample_idx) as res_sample_idx:
                        x = next(it)
                    yield res_sample_idx, x
                except StopIteration:
                    break
        finally:
            if hasattr(it, "close"):
                it.close()

    def save_state(self) -> int:
        return self._sample_index[self._rank]

    def merge_states(self, states: List[int]) -> List[int]:
        return states

    def restore_state(self, state: Optional[List[int]]) -> None:
        if state is None:
            self._sample_index = [0] * max(self.worker_config.num_workers, 1)
        else:
            assert len(state) == len(self._sample_index)
            self._sample_index = state


def get_sample_restore_key(sample: Any) -> Optional[Union[str, int]]:
    """Gets the restore key from an arbitrary sample."""
    if isinstance(sample, Sample) or hasattr(sample, "__restore_key__"):
        return sample.__restore_key__
    elif isinstance(sample, dict) and "__restore_key__" in sample:
        return sample["__restore_key__"]
    else:
        return None
