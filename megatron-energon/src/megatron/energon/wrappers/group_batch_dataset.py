# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from megatron.energon.errors import SYSTEM_EXCEPTIONS, FatalSampleError
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers._log_exception import log_exception
from megatron.energon.wrappers.base import (
    BaseSingleWrapperDataset,
    BaseSingleWrapperMergedState,
    BaseSingleWrapperState,
)
from megatron.energon.wrappers.skip import SkipSample

T_batch = TypeVar("T_batch", covariant=True)
T_batch_sample = TypeVar("T_batch_sample", covariant=True)


@dataclass
class GroupBatchState(BaseSingleWrapperState):
    batches: Any


@dataclass
class GroupBatchMergedState(BaseSingleWrapperMergedState):
    batches: List[Any]


class GroupBatchDataset(
    BaseSingleWrapperDataset[T_batch_sample, T_batch], Generic[T_batch_sample, T_batch]
):
    """This dataset wrapper transforms a dataset of samples into a dataset of batches, grouped
    by some criterion. This will not have a correct length, as it depends on the grouping.
    An example use case is: Image-Text samples, which are to be grouped by the image size into three
    size categories (e.g. 128x128, 256x256, 512x512) for efficient augmentation and batching.
    """

    dataset: SavableDataset[T_batch_sample]
    batch_size: int
    group_criterion: Callable[[T_batch_sample], Hashable]
    batcher: Callable[[List[T_batch_sample]], T_batch]
    drop_last: bool
    error_handler: Callable[[Exception, List[T_batch_sample]], None]
    worker_config: Optional[WorkerConfig]
    n_groups: int

    _state_batches: List[Optional[Dict[Hashable, List[T_batch_sample]]]]

    def __init__(
        self,
        dataset: SavableDataset[T_batch_sample],
        batch_size: int,
        group_criterion: Callable[[T_batch_sample], Hashable],
        batcher: Callable[[List[T_batch_sample]], T_batch],
        *,
        drop_last: bool = False,
        error_handler: Callable[[Exception, List[T_batch_sample]], None] = log_exception,
        worker_config: WorkerConfig,
        n_groups: int = 1,
    ):
        """Construct a GroupBatchDataset.

        Args:
            dataset: The input dataset to wrap
            batch_size: The desired batch size. The last batch may be smaller.
            group_criterion: Function which determines the group of a sample.
            batcher: Function which combines separate samples into a single object. May raise
                :exc:`megatron.energon.SkipSample` to skip a sample.
            drop_last: If True, the last batch is dropped if it is smaller than the batch size.
            error_handler: Handler for errors. Defaults to logging and ignoring the exception.
            worker_config: Configuration for the workers.
            n_groups: Number of different groups. If not set properly, `len` might be less than the
                actual number of samples yielded.
        """
        super().__init__(dataset)
        self.batch_size = batch_size
        self.group_criterion = group_criterion
        self.batcher = batcher
        self.drop_last = drop_last
        self.error_handler = error_handler
        self.worker_config = worker_config
        self.n_groups = n_groups
        self._state_batches = [{} for _ in range(max(self.worker_config.num_workers, 1))]

    def __len__(self):
        # This is only an approximation, since we don't know how many groups there are.
        # Should always overestimate (might yield less than actual number of batches).
        n_samples = len(self.dataset)
        worker_groups = self.worker_config.num_workers * self.n_groups
        n_samples_per_worker_floor = n_samples // worker_groups
        remaining_n_sample_workers = n_samples % worker_groups
        n_batches_per_worker_floor = n_samples_per_worker_floor // self.batch_size
        if n_samples_per_worker_floor % self.batch_size != 0 and not self.drop_last:
            n_batches_per_worker_floor += 1
        # Correct number of batches for the workers which yield 1 more sample (to balance)
        n_batches_per_worker_ceil = (n_samples_per_worker_floor + 1) // self.batch_size
        if n_batches_per_worker_ceil % self.batch_size != 0 and not self.drop_last:
            n_batches_per_worker_ceil += 1

        return (
            n_batches_per_worker_floor * (worker_groups - remaining_n_sample_workers)
            + n_batches_per_worker_ceil * remaining_n_sample_workers
        )

    def __iter__(self) -> Iterator[T_batch]:
        batches: Dict[Hashable, List[T_batch_sample]]
        worker_idx = self.worker_config.rank_worker_id()
        # Cleanup other states
        for i in range(self.worker_config.num_workers):
            if i != worker_idx:
                self._state_batches[i] = None
        if self._state_batches[worker_idx] is None:
            self._state_batches[worker_idx] = {}
        batches = self._state_batches[worker_idx]
        for sample in self.dataset:
            try:
                group = self.group_criterion(sample)
            except SkipSample:
                continue
            except SYSTEM_EXCEPTIONS:
                raise FatalSampleError.from_sample(sample)
            except Exception as e:
                self.error_handler(e, [sample])
                continue
            batch_group = batches.get(group)
            if batch_group is None:
                batches[group] = batch_group = []
            batch_group.append(sample)
            if len(batch_group) == self.batch_size:
                try:
                    yield self.batcher(batch_group)
                except SkipSample:
                    pass
                except SYSTEM_EXCEPTIONS:
                    raise FatalSampleError.from_sample(batch_group)
                except Exception as e:
                    self.error_handler(e, batch_group)
                del batches[group]
        if not self.drop_last:
            for batch in batches.values():
                try:
                    yield self.batcher(batch)
                except SkipSample:
                    pass
                except SYSTEM_EXCEPTIONS:
                    raise FatalSampleError.from_sample(batch)
                except Exception as e:
                    self.error_handler(e, batch)
        self._state_batches[worker_idx] = None

    def save_state(self) -> GroupBatchState:
        return GroupBatchState.extend(
            super().save_state(),
            batches=dict(self._state_batches[self.worker_config.rank_worker_id()]),
        )

    def merge_states(self, states: List[GroupBatchState]) -> GroupBatchMergedState:
        assert all(s is None or isinstance(s, GroupBatchState) for s in states)
        return GroupBatchMergedState.extend(
            super().merge_states(states),
            batches=[{} if s is None else s.batches for s in states],
        )

    def restore_state(self, state: GroupBatchMergedState) -> None:
        super().restore_state(state)
        if state is None:
            self._state_batches = [{} for _ in range(max(self.worker_config.num_workers, 1))]
        else:
            assert isinstance(state, GroupBatchMergedState)
            self._state_batches = state.batches

    def can_restore_sample(self) -> bool:
        return False

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_batch:
        # TODO: We'd need to store multiple indices to restore a batch
        # Also, returned elements don't support __restore_key__. Would need extension.
        raise NotImplementedError("GroupBatchDataset does not support random access.")

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "batch_size": self.batch_size,
            "group_criterion": self._function_config(self.group_criterion),
            "batcher": self._function_config(self.batcher),
            "drop_last": self.drop_last,
            "error_handler": self._function_config(self.error_handler),
            "worker_config": self.worker_config.config(),
            "n_groups": self.n_groups,
        }

    def __str__(self):
        return f"GroupBatchDataset(batch_size={self.batch_size}, group_criterion={self.group_criterion}, batcher={self.batcher}, drop_last={self.drop_last}, error_handler={self.error_handler}, worker_config={self.worker_config}, n_groups={self.n_groups}, dataset={self.dataset})"
