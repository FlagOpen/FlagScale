# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterator, List, Literal, Optional, TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import (
    BaseSingleWrapperDataset,
    BaseSingleWrapperMergedState,
    BaseSingleWrapperState,
)

T_sample = TypeVar("T_sample")


def default_get_keys(batch: Any) -> Optional[List[str]]:
    """Default get_keys, which has some heuristics to find the sample keys."""
    if isinstance(batch, list):
        batch = batch[0]
    if (
        hasattr(batch, "__key__")
        and isinstance(batch.__key__, list)
        and all(isinstance(k, str) for k in batch.__key__)
    ):
        return batch.__key__
    elif (
        hasattr(batch, "__keys__")
        and isinstance(batch.__keys__, list)
        and all(isinstance(k, str) for k in batch.__keys__)
    ):
        return batch.__keys__
    elif (
        isinstance(batch, dict)
        and "__key__" in batch
        and all(isinstance(k, str) for k in batch["__key__"])
    ):
        return batch["__key__"]
    elif (
        isinstance(batch, dict)
        and "__keys__" in batch
        and all(isinstance(k, str) for k in batch["__keys__"])
    ):
        return batch["__keys__"]
    elif (
        isinstance(batch, dict)
        and "keys" in batch
        and all(isinstance(k, str) for k in batch["keys"])
    ):
        return batch["keys"]
    return None


@dataclass
class LogSampleState(BaseSingleWrapperState):
    step: int


@dataclass
class LogSampleMergedState(BaseSingleWrapperMergedState):
    step: List[int]


class LogSampleDataset(BaseSingleWrapperDataset[T_sample, T_sample], Generic[T_sample]):
    """This dataset logs every yielded sample to the debug logs."""

    get_keys_fn: Callable[[T_sample], Optional[List[str]]]
    mode: Literal["train", "val"]
    worker_config: WorkerConfig
    _step: List[int]

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        mode: Literal["train", "val"],
        worker_config: WorkerConfig,
        get_keys_fn: Callable[[T_sample], Optional[List[str]]] = default_get_keys,
    ):
        """Construct the log sample dataset, which logs every yielded sample to the debug logs.

        Args:
            dataset: The input dataset to wrap
        """
        super().__init__(dataset)
        self.get_keys_fn = get_keys_fn
        self.mode = mode
        self.worker_config = worker_config
        self._step = [0] * max(self.worker_config.num_workers, 1)

    def __len__(self):
        return len(self.dataset)

    def _log(self, sample: T_sample) -> None:
        if self.worker_config.should_log(level=1):
            log_entry = {
                "t": "yield_batch",
                "r": self.worker_config.rank,
                "w": self.worker_config.global_worker_id(),
                "m": self.mode,
                "idx": self._step[self.worker_config.rank_worker_id()],
            }
            keys = self.get_keys_fn(sample)
            if keys is not None:
                log_entry["keys"] = keys

            self.worker_config.worker_log(log_entry)

    def __iter__(self) -> Iterator[T_sample]:
        worker_id = self.worker_config.rank_worker_id()
        for sample in self.dataset:
            self._log(sample)
            self._step[worker_id] += 1
            yield sample

    def save_state(self) -> LogSampleState:
        return LogSampleState.extend(
            super().save_state(),
            step=self._step[self.worker_config.rank_worker_id()],
        )

    def merge_states(self, states: List[Optional[LogSampleState]]) -> LogSampleMergedState:
        assert all(s is None or isinstance(s, LogSampleState) for s in states)
        return LogSampleMergedState.extend(
            super().merge_states(states),
            step=[0 if state is None else state.step for state in states],
        )

    def restore_state(self, state: Optional[LogSampleMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._step = [0] * max(self.worker_config.num_workers, 1)
        else:
            assert isinstance(state, LogSampleMergedState)
            self._step = state.step

    def config(self) -> Dict[str, Any]:
        # Transparent logger, it won't change the samples
        return self.dataset.config()

    def __str__(self):
        return f"LogSampleDataset(mode={self.mode}, get_keys_fn={self.get_keys_fn}, dataset={self.dataset})"
