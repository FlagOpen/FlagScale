# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar, Union

from megatron.energon.flavors.base_dataset import (
    MergedState,
    SavableDataset,
    State,
    add_sample_restore_key,
)
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseWrapperDataset

T_sample = TypeVar("T_sample")


@dataclass
class ConcatState(State):
    #: State of the inner datasets
    dataset_states: List[State]


@dataclass
class ConcatMergedState(MergedState):
    #: State of the inner datasets
    dataset_states: List[MergedState]


class ConcatDataset(BaseWrapperDataset[T_sample], Generic[T_sample]):
    """
    This dataset wrapper concatenates multiple iterable datasets together. The datasets must be
    finite, otherwise not all datasets can be sampled. This is only useful for validation / test
    datasets.
    """

    datasets: Tuple[SavableDataset[T_sample], ...]

    def __init__(
        self,
        *datasets: SavableDataset[T_sample],
        worker_config: WorkerConfig,
    ):
        """Construct a concatenated dataset."""
        super().__init__()
        self.worker_config = worker_config
        self.datasets = datasets
        assert len(self) >= 0, "Datasets must be finite."

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __iter__(self) -> Iterator[T_sample]:
        for ds_idx, dataset in enumerate(self.datasets):
            for sample in dataset:
                yield add_sample_restore_key(
                    sample,
                    ds_idx,
                    src=self,
                )

    def worker_has_samples(self) -> bool:
        return any(dataset.worker_has_samples() for dataset in self.datasets)

    def save_state(self) -> ConcatState:
        return ConcatState(
            dataset_states=[dataset.save_state() for dataset in self.datasets],
        )

    def merge_states(self, states: List[ConcatState]) -> ConcatMergedState:
        assert all(s is None or isinstance(s, ConcatState) for s in states)
        assert all(s is None or len(s.dataset_states) == len(self.datasets) for s in states)
        return ConcatMergedState(
            dataset_states=[
                dataset.merge_states(
                    [None if s is None else s.dataset_states[ds_idx] for s in states]
                )
                for ds_idx, dataset in enumerate(self.datasets)
            ],
        )

    def restore_state(self, state: Optional[ConcatMergedState]) -> None:
        if state is None:
            for dataset in self.datasets:
                dataset.restore_state(None)
        else:
            assert isinstance(state, ConcatMergedState)
            assert len(self.datasets) == len(state.dataset_states)
            for dataset, dstate in zip(self.datasets, state.dataset_states):
                dataset.restore_state(dstate)

    def can_restore_sample(self) -> bool:
        return all(dataset.can_restore_sample() for dataset in self.datasets)
    
    def assert_can_restore(self) -> None:
        for dataset in self.datasets:
            dataset.assert_can_restore()

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        id, ds_idx = index[:2]
        assert id == type(self).__name__
        index = index[2:]
        assert isinstance(ds_idx, int)
        return add_sample_restore_key(
            self.datasets[ds_idx].restore_sample(index),
            ds_idx,
            src=self,
        )

    def verify_worker_config(self, worker_config: WorkerConfig) -> None:
        super().verify_worker_config(worker_config)
        for dataset in self.datasets:
            dataset.verify_worker_config(worker_config)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "datasets": [dataset.config() for dataset in self.datasets],
        }

    def __str__(self):
        return f"ConcatDataset(datasets={self.datasets})"
