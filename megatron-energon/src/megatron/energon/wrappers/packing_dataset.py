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
from megatron.energon.flavors.base_dataset import SavableDataset, set_sample_restore_key
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers._log_exception import log_exception
from megatron.energon.wrappers.base import (
    BaseSingleWrapperDataset,
    BaseSingleWrapperMergedState,
    BaseSingleWrapperState,
    SampleIndex,
    get_sample_restore_key,
)
from megatron.energon.wrappers.buffer import (
    SampleBufferMergedState,
    SampleBufferState,
    SavableSampleBuffer,
)
from megatron.energon.wrappers.skip import SkipSample

T_sample = TypeVar("T_sample")
T_batch_sample = TypeVar("T_batch_sample")


@dataclass
class PackingState(BaseSingleWrapperState):
    reading_buffer: SampleBufferState
    pre_packing_buffer: SampleBufferState
    pre_packing_lengths: List[int]
    pre_packing_index: int
    final_packing_index: int


@dataclass
class PackingMergedState(BaseSingleWrapperMergedState):
    reading_buffer: SampleBufferMergedState
    pre_packing_buffer: SampleBufferMergedState
    pre_packing_lengths: List[List[int]]
    pre_packing_index: List[int]
    final_packing_index: List[int]


class PackingDataset(
    BaseSingleWrapperDataset[T_sample, T_batch_sample], Generic[T_sample, T_batch_sample]
):
    """This dataset wrapper transforms samples of a dataset into chunks/packs of samples, which are
    then combined into a batch."""

    buffer_size: int
    pre_packer: Callable[[List[T_sample]], List[List[T_sample]]]
    final_packer: Callable[[List[T_sample]], T_batch_sample]
    final_packer_stateless: bool
    error_handler: Callable[[Exception, List[T_sample]], None]
    worker_config: WorkerConfig

    #: The buffer for collecting the samples that shall be packed.
    _reading_buffer: SavableSampleBuffer

    #: Contains the pre-selected samples to be packed.
    #: The full buffer will be passed to the pre_packer.
    _pre_packing_buffer: SavableSampleBuffer

    #: Lengths of the selected groups of samples to be packed together.
    #: The samples are stored sequentially in the pre_packing_buffer because
    #: SavableSampleBuffer doesn't support nesting. But to keep the groups
    #: separate, we need to store the lengths of the groups here.
    _pre_packing_lengths: List[List[int]]

    #: Sample index for the pre_packer
    _pre_packing_sample_index: SampleIndex

    #: Sample index for the final_packer
    _final_packing_sample_index: SampleIndex

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        buffer_size: int,
        pre_packer: Callable[[List[T_sample]], List[List[T_sample]]],
        final_packer: Callable[[List[T_sample]], T_batch_sample],
        *,
        final_packer_stateless: bool = False,
        error_handler: Callable[[Exception, List[T_sample]], None] = log_exception,
        worker_config: WorkerConfig,
    ):
        """Construct a PackingDataset which is used for sequence packing.
        Using a pre_packer and final_packer, it buffers the incoming samples, groups
        them together based on the logic provided by the pre_packer, and then (using
        the final_packer) combines each group into a packed single sample also called
        a "pack" or a "packed sequence".

        Args:
            dataset: The input dataset to wrap
            buffer_size: The desired size of the input buffer for pre packing. Last buffer of a dataset may be smaller.
            pre_packer: Function which selects samples from the buffer to be packed together.
                May raise :exc:`megatron.energon.SkipSample` to skip a buffer.
            final_packer: Function which combines the selected samples into a single sample.
            final_packer_stateless: If True, the final_packer is stateless, thus samples can be
                stored/restored.
            error_handler: Function which handles exceptions raised by the batcher. The default
                implementation logs the exception.
            worker_config: Configuration for the workers.
        """
        super().__init__(dataset)

        assert buffer_size > 0, "Packing buffer size must be greater than 0."

        self.buffer_size = buffer_size
        self.pre_packer = pre_packer
        self.final_packer = final_packer
        self.final_packer_stateless = final_packer_stateless
        self.error_handler = error_handler
        self.worker_config = worker_config
        self._reading_buffer = SavableSampleBuffer(dataset, worker_config)
        self._pre_packing_buffer = SavableSampleBuffer(dataset, worker_config)
        self._pre_packing_lengths = [[] for _ in range(max(worker_config.num_workers, 1))]
        self._pre_packing_sample_index = SampleIndex(worker_config, src=self)
        self._final_packing_sample_index = SampleIndex(worker_config, src=self)

    def __len__(self):
        """The real length is unknown, since it depends on the packing function.
        We approximate it by the length of the source dataset."""

        return len(self.dataset)

    def _fill_reading_buffer(self, source_iter: Iterator) -> bool:
        """
        Fill the reading buffer with samples from the dataset source iterator.

        Args:
            source_iter: Iterator of samples from the dataset.

        Returns:
            True if samples are successfully read into the buffer, False if no more data.
        """

        while len(self._reading_buffer) + len(self._pre_packing_buffer) < self.buffer_size:
            try:
                sample = next(source_iter)
                self._reading_buffer.append(sample)
            except StopIteration:
                return False
        return True

    def __iter__(self) -> Iterator[T_sample]:
        worker_idx = self.worker_config.rank_worker_id()
        pre_packing_lengths = self._pre_packing_lengths[worker_idx]
        # The source dataset
        src_iter = iter(self.dataset)

        self._pre_packing_buffer.worker_start()
        self._reading_buffer.worker_start()

        def next_pre_pack():
            """Take the samples from the reading buffer and select groups of samples to be packed
            together."""

            assert len(self._pre_packing_buffer) == 0
            if len(self._reading_buffer) > 0:
                # Take all samples from the reading buffer and pre_pack them
                samples = list(self._reading_buffer)
                # Clear buffer and pre_packing_lengths
                self._reading_buffer.clear()
                pre_packing_lengths.clear()
                # Now pre pack the samples
                try:
                    with self._pre_packing_sample_index.ctx():
                        pre_packs = self.pre_packer(samples)
                except SkipSample:
                    pre_packs = []
                except SYSTEM_EXCEPTIONS:
                    raise FatalSampleError.from_sample(samples)
                except Exception as e:
                    self.error_handler(e, samples)
                    pre_packs = []

                # Put the pre-packed samples into the pre_packing_buffer
                # They will be flattened here to avoid nested buffers
                # But the lengths of the groups are stored in pre_packing_lengths
                # so that the groups can be separated later
                for pre_pack in pre_packs:
                    self._pre_packing_buffer.extend(pre_pack)
                    pre_packing_lengths.append(len(pre_pack))

        def next_final_pack():
            """Yield the next packs from the buffer. The final packer is called on the fly."""

            pack = list(self._pre_packing_buffer[: pre_packing_lengths[0]])
            del self._pre_packing_buffer[: pre_packing_lengths[0]]
            del pre_packing_lengths[0]
            try:
                pack_restore_keys = tuple(get_sample_restore_key(sample) for sample in pack)
                with self._final_packing_sample_index.ctx() as pack_idx:
                    final_packed_sample = self.final_packer(pack)
                if isinstance(final_packed_sample, Generator):
                    assert inspect.isgeneratorfunction(
                        self.final_packer
                    ), f"Generator in {self.map_fn} but not marked as such."
                    for pack_sub_idx, (pack_idx, inner_batch_sample) in enumerate(
                        self._final_packing_sample_index.iter_ctx(final_packed_sample, pack_idx)
                    ):
                        yield set_sample_restore_key(
                            inner_batch_sample,
                            pack_idx,
                            pack_sub_idx,
                            *pack_restore_keys,
                            src=self,
                        )
                else:
                    yield set_sample_restore_key(
                        final_packed_sample,
                        pack_idx,
                        *pack_restore_keys,
                        src=self,
                    )
            except SkipSample:
                pass
            except SYSTEM_EXCEPTIONS:
                raise FatalSampleError.from_sample(pack)
            except Exception as e:
                self.error_handler(e, pack)

        # Main loop:
        stopping = False
        while not stopping:

            pre_pack_round = 0
            while len(pre_packing_lengths) == 0:
                # Fill a portion of the buffer
                if not self._fill_reading_buffer(src_iter):
                    # Break out of both loops when the source is exhausted.
                    # But yield the remaining packs first.
                    if len(self._reading_buffer) > 0:
                        next_pre_pack()
                    stopping = True
                    break

                assert len(self._pre_packing_buffer) == 0
                assert len(self._reading_buffer) == self.buffer_size

                next_pre_pack()

                pre_pack_round += 1
                if pre_pack_round > 10:
                    raise RuntimeError("Pre packer did not yield any packs after 10 rounds.")

            if not stopping:
                yield from next_final_pack()
            else:
                break

        # Yield the remaining packs, flushing the collecting buffer
        while len(pre_packing_lengths) > 0:
            yield from next_final_pack()

    def save_state(self) -> PackingState:
        return PackingState.extend(
            super().save_state(),
            reading_buffer=self._reading_buffer.save_state(),
            pre_packing_buffer=self._pre_packing_buffer.save_state(),
            pre_packing_lengths=list(
                self._pre_packing_lengths[self.worker_config.rank_worker_id()]
            ),
            final_packing_index=self._final_packing_sample_index.save_state(),
            pre_packing_index=self._pre_packing_sample_index.save_state(),
        )

    def merge_states(self, states: List[PackingState]) -> PackingMergedState:
        assert all(s is None or isinstance(s, PackingState) for s in states)
        return PackingMergedState.extend(
            super().merge_states(states),
            reading_buffer=self._reading_buffer.merge_states(
                [None if s is None else s.reading_buffer for s in states]
            ),
            pre_packing_buffer=self._pre_packing_buffer.merge_states(
                [None if s is None else s.pre_packing_buffer for s in states]
            ),
            pre_packing_lengths=[[0, 0] if s is None else s.pre_packing_lengths for s in states],
            final_packing_index=self._final_packing_sample_index.merge_states(
                [0 if state is None else state.final_packing_index for state in states]
            ),
            pre_packing_index=self._pre_packing_sample_index.merge_states(
                [0 if state is None else state.pre_packing_index for state in states]
            ),
        )

    def restore_state(self, state: Optional[PackingMergedState]) -> None:
        super().restore_state(state)
        if state is None:
            self._reading_buffer.restore_state(None)
            self._pre_packing_buffer.restore_state(None)
            self._pre_packing_lengths = [[] for _ in range(max(self.worker_config.num_workers, 1))]
            self._pre_packing_sample_index.restore_state(None)
            self._final_packing_sample_index.restore_state(None)
        else:
            assert isinstance(state, PackingMergedState)
            self._reading_buffer.restore_state(state.reading_buffer)
            self._pre_packing_buffer.restore_state(state.pre_packing_buffer)
            self._pre_packing_lengths = state.pre_packing_lengths
            self._pre_packing_sample_index.restore_state(state.pre_packing_index)
            self._final_packing_sample_index.restore_state(state.final_packing_index)

    def can_restore_sample(self) -> bool:
        # Cannot really verify if the returned elements contain a __restore_key__.
        # If the user wants to use this, well...
        return self.final_packer_stateless and self.dataset.can_restore_sample()
    
    def assert_can_restore(self):
        assert self.final_packer_stateless, f"Final packer {self.final_packer} must be stateless to restore samples."
        self.dataset.assert_can_restore()

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        # We need to store multiple indices to restore a batch.
        self.assert_can_restore()
        if inspect.isgeneratorfunction(self.final_packer):
            id, pack_idx, pack_sub_idx, *pack_restore_keys = index
            assert id == type(self).__name__
        else:
            id, pack_idx, *pack_restore_keys = index
            assert id == type(self).__name__
        batch = [self.dataset.restore_sample(inner_idx) for inner_idx in pack_restore_keys]
        with self._final_packing_sample_index.ctx(pack_idx):
            final_pack = self.final_packer(batch)
        if isinstance(final_pack, Generator):
            assert inspect.isgeneratorfunction(
                self.final_packer
            ), f"Generator in {self.map_fn} but not marked as such."
            for cur_batch_sub_idx, (pack_idx, inner_batch_sample) in enumerate(
                self._final_packing_sample_index.iter_ctx(final_pack, pack_idx)
            ):
                if cur_batch_sub_idx == pack_sub_idx:
                    return set_sample_restore_key(
                        inner_batch_sample,
                        pack_idx,
                        pack_sub_idx,
                        *pack_restore_keys,
                        src=self,
                    )
            assert False, f"Pack sub-index {pack_sub_idx} not found in pack"
        else:
            return set_sample_restore_key(final_pack, pack_idx, *pack_restore_keys, src=self)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "buffer_size": self.buffer_size,
            "pre_packer": self._function_config(self.pre_packer),
            "final_packer": self._function_config(self.final_packer),
            "final_packer_stateless": self.final_packer_stateless,
            "error_handler": self._function_config(self.error_handler),
            "worker_config": self.worker_config.config(),
            "dataset": self.dataset.config(),
        }

    def __str__(self):
        return f"PackingDataset(buffer_size={self.buffer_size}, pre_packer={self.pre_packer}, final_packer={self.final_packer}, dataset={self.dataset})"
