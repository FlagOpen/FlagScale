# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from megatron.energon.errors import SYSTEM_EXCEPTIONS, FatalSampleError
from megatron.energon.flavors.base_dataset import SavableDataset, add_sample_restore_key
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers._log_exception import log_exception
from megatron.energon.wrappers.base import (
    BaseSingleWrapperDataset,
    BaseSingleWrapperMergedState,
    BaseSingleWrapperState,
    SampleIndex,
    get_sample_restore_key,
)
from megatron.energon.wrappers.skip import SkipSample

T_sample = TypeVar("T_sample")
T_sample_out = TypeVar("T_sample_out")


@dataclass
class MapState(BaseSingleWrapperState):
    sample_index: int


@dataclass
class MapMergedState(BaseSingleWrapperMergedState):
    sample_indexes: List[int]


@dataclass
class MapGeneratorState(MapState):
    generator_sample_key: Any
    generator_offset: int


@dataclass
class MapGeneratorMergedState(MapMergedState):
    generator_sample_keys: List[Any]
    generator_offsets: List[int]


class MapDataset(BaseSingleWrapperDataset[T_sample, T_sample_out], Generic[T_sample, T_sample_out]):
    """This dataset wrapper applies a custom function to transform each sample."""

    map_fn: Callable[[T_sample], Union[T_sample_out, Generator[T_sample_out, None, None]]]
    error_handler: Callable[[Exception, T_sample], None]
    stateless_map_fn: bool
    _sample_index: SampleIndex
    _generator_sample_keys: List[Optional[Any]]
    _generator_offsets: List[Optional[int]]

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        map_fn: Callable[[T_sample], Union[T_sample_out, Generator[T_sample_out, None, None]]],
        *,
        error_handler: Callable[[Exception, T_sample], None] = log_exception,
        stateless_map_fn: bool = False,
        worker_config: WorkerConfig,
    ):
        """Construct a MapDataset.

        If this should be savable, the map_fn must only return a sample, or a generator yielding
        0 or 1 sample per input sample. Otherwise this will be broken (see `IterMapDataset`).

        Args:
            dataset: The input dataset to wrap
            map_fn: The function to apply to each sample. May raise
                :exc:`megatron.energon.SkipSample` to skip a sample. Alternatively, may return a
                generator to yield multiple or no samples.
            error_handler: Handler for errors. Defaults to logging and ignoring the exception.
            stateless_map_fn: If true, the map_fn is deterministic and stateless
                (thus key for random access can propagate to inner dataset). Defaults to False.
        """
        super().__init__(dataset)
        self.map_fn = map_fn
        self.error_handler = error_handler
        self.stateless_map_fn = stateless_map_fn
        self.worker_config = worker_config
        self._sample_index = SampleIndex(worker_config, src=self)
        self._generator_sample_keys = [None] * max(self.worker_config.num_workers, 1)
        self._generator_offsets = [None] * max(self.worker_config.num_workers, 1)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_sample_out]:
        worker_id = self.worker_config.rank_worker_id()
        if self._generator_sample_keys[worker_id] is not None:
            assert self._generator_offsets[worker_id] is not None
            sample = self.dataset.restore_sample(self._generator_sample_keys[worker_id])
            # Do not increment the sample index, use previous index
            with self._sample_index.ctx(self._sample_index.current_idx) as sample_idx:
                mapped_sample = self.map_fn(sample)
            assert isinstance(mapped_sample, Generator)
            assert inspect.isgeneratorfunction(
                self.map_fn
            ), f"Generator in {self.map_fn} but not marked as such."
            target_offset = self._generator_offsets[worker_id]
            self._generator_offsets[worker_id] = 0
            for idx, (sample_idx, inner_sample) in enumerate(
                self._sample_index.iter_ctx(mapped_sample, sample_idx)
            ):
                # Skip other samples
                if idx >= target_offset:
                    self._generator_offsets[worker_id] = idx + 1
                    yield add_sample_restore_key(
                        inner_sample,
                        sample_idx,
                        idx,
                        src=self,
                    )
            self._generator_sample_keys[worker_id] = None
            self._generator_offsets[worker_id] = None

        for sample in self.dataset:
            try:
                with self._sample_index.ctx() as sample_idx:
                    mapped_sample = self.map_fn(sample)
                if isinstance(mapped_sample, Generator):
                    assert inspect.isgeneratorfunction(
                        self.map_fn
                    ), f"Generator in {self.map_fn} but not marked as such."
                    self._generator_sample_keys[worker_id] = get_sample_restore_key(sample)
                    self._generator_offsets[worker_id] = 0
                    # In case of a generator, additionally store the index of the yielded samples
                    # per input sample
                    for idx, (sample_idx, inner_sample) in enumerate(
                        self._sample_index.iter_ctx(mapped_sample, sample_idx)
                    ):
                        self._generator_offsets[worker_id] = idx + 1
                        yield add_sample_restore_key(
                            inner_sample,
                            sample_idx,
                            idx,
                            src=self,
                        )
                    self._generator_sample_keys[worker_id] = None
                    self._generator_offsets[worker_id] = None
                else:
                    yield add_sample_restore_key(
                        mapped_sample,
                        sample_idx,
                        src=self,
                    )
            except SkipSample:
                pass
            except SYSTEM_EXCEPTIONS as e:
                raise FatalSampleError.from_sample(sample)
            except Exception as e:
                self.error_handler(e, sample)

    def save_state(self) -> MapState:
        state = MapState.extend(
            super().save_state(),
            sample_index=self._sample_index.save_state(),
        )
        if self._generator_offsets[self.worker_config.rank_worker_id()] is not None:
            state = MapGeneratorState.extend(
                state,
                generator_sample_key=self._generator_sample_keys[
                    self.worker_config.rank_worker_id()
                ],
                generator_offset=self._generator_offsets[self.worker_config.rank_worker_id()],
            )
        return state

    def merge_states(self, states: List[MapState]) -> MapMergedState:
        assert all(s is None or isinstance(s, MapState) for s in states)
        state = MapMergedState.extend(
            super().merge_states(states),
            sample_indexes=self._sample_index.merge_states(
                [0 if state is None else state.sample_index for state in states]
            ),
        )
        if any(isinstance(s, MapGeneratorState) for s in states):
            state = MapGeneratorMergedState.extend(
                state,
                generator_sample_keys=[
                    state.generator_sample_key if isinstance(state, MapGeneratorState) else None
                    for state in states
                ],
                generator_offsets=[
                    state.generator_offset if isinstance(state, MapGeneratorState) else None
                    for state in states
                ],
            )
        return state

    def restore_state(self, state: Optional[MapMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._sample_index.restore_state(None)
            self._generator_sample_keys = [None] * max(self.worker_config.num_workers, 1)
            self._generator_offsets = [None] * max(self.worker_config.num_workers, 1)
        else:
            assert isinstance(state, MapMergedState)
            self._sample_index.restore_state(state.sample_indexes)
            if isinstance(state, MapGeneratorMergedState):
                self._generator_sample_keys = state.generator_sample_keys
                self._generator_offsets = state.generator_offsets
            else:
                self._generator_sample_keys = [None] * max(self.worker_config.num_workers, 1)
                self._generator_offsets = [None] * max(self.worker_config.num_workers, 1)

    def can_restore_sample(self) -> bool:
        return self.stateless_map_fn and self.dataset.can_restore_sample()
    
    def assert_can_restore(self) -> None:
        assert self.stateless_map_fn, f"MapDataset can only restore samples if map_fn {self.map_fn} is stateless."
        self.dataset.assert_can_restore()

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample_out:
        self.assert_can_restore()
        if inspect.isgeneratorfunction(self.map_fn):
            id, sample_idx, local_idx = index[:3]
            assert id == type(self).__name__
            index = index[3:]
            assert isinstance(local_idx, int)
        else:
            id, sample_idx = index[:2]
            assert id == type(self).__name__
            index = index[2:]
        inner_sample = self.dataset.restore_sample(index)
        with self._sample_index.ctx(sample_idx):
            mapped_sample = self.map_fn(inner_sample)
        if isinstance(mapped_sample, Generator):
            assert inspect.isgeneratorfunction(
                self.map_fn
            ), f"Generator in {self.map_fn} but not marked as such."
            for idx, (sample_idx, res_sample) in enumerate(
                self._sample_index.iter_ctx(mapped_sample, sample_idx)
            ):
                if idx == local_idx:
                    return add_sample_restore_key(res_sample, sample_idx, local_idx, src=self)
            assert (
                False
            ), "Generator did not yield enough samples, but is marked stateless/deterministic."
        else:
            return add_sample_restore_key(mapped_sample, sample_idx, src=self)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
            "map_fn": self._function_config(self.map_fn),
        }

    def __str__(self):
        return f"MapDataset(map_fn={self.map_fn}, dataset={self.dataset})"
