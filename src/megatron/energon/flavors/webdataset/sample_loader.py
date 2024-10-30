# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import tarfile
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Set, Tuple, Union

import torch

from megatron.energon.epathlib import EPath
from megatron.energon.errors import SYSTEM_EXCEPTIONS, FatalSampleError
from megatron.energon.flavors.base_dataset import MergedState, SavableDataset, State
from megatron.energon.flavors.webdataset.config import skip_meta_re, split_name_re
from megatron.energon.flavors.webdataset.itar import get_itar_byte_offset, open_itar
from megatron.energon.flavors.webdataset.structs import FilteredSample, ShardInfo, reraise_exception
from megatron.energon.rng import WorkerRng, WorkerRngMergedState, WorkerRngState
from megatron.energon.worker import WorkerConfig


@dataclass
class ShardState:
    """
    Represents the iteration state of a single shard. I.e. contains a reference to the shard and the
    reader's offset.
    """

    #: The shard of this state
    shard: ShardInfo
    #: Sample offset relative to the sub-shard file offset
    offset: int
    #: Byte offset relative to the sub-shard file byte offset
    byte_offset: int


@dataclass
class SampleLoaderState(State):
    """
    The savable state for the wds sample loader. Contains the active and pending shards.
    """

    #: Rng state
    rng: WorkerRngState
    #: Pending shards are the shards which have not yet been opened, but should be processed
    # in the current "epoch"
    pending_shards: Optional[List[ShardInfo]]
    #: The active shards are the currently opened shards. May contain `None`, if there are fewer
    # shards available (i.e. pending_shards empty) than parallel shard iterators requested.
    active_shards: Optional[List[Optional[ShardState]]]
    #: The total number of samples retrieved, it's just a monotonically increasing counter
    sample_count: int
    #: Number of epochs this dataset has been iterated over
    epoch_count: int
    #: Number of samples retrieved in current epoch
    epoch_sample_count: int


@dataclass
class SampleLoaderMergedState(MergedState):
    #: Rng state
    rng: WorkerRngMergedState
    #: Pending shards are the shards which have not yet been opened, but should be processed
    # in the current "epoch"
    pending_shards: List[Optional[List[ShardInfo]]]
    #: The active shards are the currently opened shards. May contain `None`, if there are fewer
    # shards available (i.e. pending_shards empty) than parallel shard iterators requested.
    active_shards: List[Optional[List[Optional[ShardState]]]]
    #: The total number of samples retrieved, it's just a monotonically increasing counter
    sample_count: List[int]
    #: Number of epochs this dataset has been iterated over
    epoch_count: List[int]
    #: The number of samples retrieved in current epoch
    epoch_sample_count: List[int]


class WebdatasetSampleLoaderDataset(SavableDataset[FilteredSample]):
    """Internal class for loading samples from webdataset shards"""

    #: All shards for all workers `shards[worker_idx][shard_idx]`
    shards: List[List[ShardInfo]]
    #: All shards for all workers accessible by name and offset
    # `shards[worker_idx][(shard_name, offset)]`, created lazily
    _shards_by_key: Optional[List[Dict[Tuple[str, int], ShardInfo]]] = None
    #: Paths to shards for all workers by name `shards[worker_idx][shard_name]`, created lazily
    _shard_paths_by_name: Optional[Dict[str, EPath]] = None
    #: The data parallel worker config
    worker_config: WorkerConfig
    # Sample keys to ignore
    exclude: Set[str]
    # File extensions to load (or None to load all)
    extensions: Optional[Set[str]]

    # If true, loop the shards
    loop: bool
    # If = 1, every sample is seen exactly once per epoch. If > 1, samples
    # (or rather shard slices) are shuffled within this number of epochs (i.e. randomly
    # selected without replacement). If None, the shards are effectively shuffle over
    # infinite epochs (i.e. shard slices are drawn with replacement).
    shuffle_over_epochs: Optional[int]
    # Number of parallel iterators to be opened simultaneously (and random sample between them)
    parallel_shard_iters: int
    # Error handler
    handler: Callable[[Exception, str], None]

    # Worker's random generator
    _worker_rng: WorkerRng
    #: Pending shards are the shards which have not yet been opened, but should be processed
    # in the current "epoch"
    _pending_shards: List[List[ShardInfo]]
    #: The active shards are the currently opened shards. May contain `None`, if there are fewer
    # shards available (i.e. pending_shards empty) than parallel shard iterators requested.
    _active_shards_state: List[Optional[List[Optional[ShardState]]]]
    #: The total number of samples retrieved, it's just a monotonically increasing counter
    _sample_count: List[int]
    #: Number of epochs this dataset has been iterated over
    _epoch_count: List[int]
    #: The number of samples retrieved in current epoch
    _epoch_sample_count: List[int]

    def __init__(
        self,
        rank_shards: List[List[ShardInfo]],
        *,
        worker_config: WorkerConfig,
        exclude: Set[str],
        part_filter: Optional[Callable[[str], bool]] = None,
        loop: bool = False,
        shuffle_over_epochs: Optional[int] = None,
        parallel_shard_iters: int = 1,
        handler: Callable[[Exception, str], None] = reraise_exception,
    ):
        """
        The webdataset loader. Iterates over the shard infos and yields the samples.

        Args:
            rank_shards: The shards to iterate over for each worker of the current rank.
            worker_config: The worker configuration.
            exclude: A set of strings of the form "<shard name>" or "<shard name>/<sample index>" to
                exclude from iteration.
            part_filter: If not None, use this function to filter out wds files.
            loop: If true, loop the shards indefinitely.
            shuffle_over_epochs: If None, disable shuffling.
                If = 1, every sample is seen exactly once per epoch.
                If > 1, samples (or rather shard slices) are shuffled within this number of epochs
                (i.e. randomly selected without replacement).
                If -1, the shards are effectively shuffle over infinite epochs (i.e. shard slices
                are drawn with replacement).
            parallel_shard_iters: If > 1, samples are randomly drawn from parallel shard iterators.
                This will not impact performance, but increase randomness. If = 1, the shards are
                iterated in order.
            handler: Exception handler. Args: (exception, key).
        """
        super().__init__()
        self.shards = rank_shards
        self.worker_config = worker_config
        self.exclude = exclude
        self.part_filter = part_filter
        self.loop = loop
        self.shuffle_over_epochs = shuffle_over_epochs
        self.parallel_shard_iters = parallel_shard_iters
        self.handler = handler
        self._worker_rng = WorkerRng(worker_config)
        self._pending_shards = [[] for _ in range(len(self.shards))]
        self._active_shards_state = [[None] * parallel_shard_iters for _ in range(len(self.shards))]
        self._sample_count = [0] * len(self.shards)
        self._epoch_count = [0] * len(self.shards)
        self._epoch_sample_count = [0] * len(self.shards)
        assert shuffle_over_epochs is None or shuffle_over_epochs == -1 or shuffle_over_epochs >= 1
        assert self.parallel_shard_iters >= 1

    @property
    def shards_by_key(self) -> List[Dict[Tuple[str, int], ShardInfo]]:
        if self._shards_by_key is None:
            self._shards_by_key = [
                {(shard.name, shard.offset): shard for shard in shards} for shards in self.shards
            ]
        return self._shards_by_key

    @property
    def shard_path_map(self) -> Dict[str, EPath]:
        if self._shard_paths_by_name is None:
            self._shard_paths_by_name = {
                shard.name: shard.path for shards in self.shards for shard in shards
            }
        return self._shard_paths_by_name

    def _shards_once(self, shards: List[ShardInfo]) -> List[ShardInfo]:
        """Possibly (re)shuffles the shards using the random generator."""
        if self.shuffle_over_epochs is None:
            # No shuffling
            return list(shards)
        elif self.shuffle_over_epochs == -1:
            # Shuffle with replacement (i.e. infinite epochs), effectively return as many shards
            # as are required for parallel shard iterators.
            # Next shards are drawn in the _shards_iter function.
            return [
                shards[self._worker_rng.randbelow(len(shards))]
                for _ in range(self.parallel_shard_iters)
            ]
        elif self.shuffle_over_epochs >= 1:
            # Shuffle without replacement (potentially over multiple epochs)
            return self._worker_rng.shuffle(shards * self.shuffle_over_epochs)
        else:
            raise ValueError(f"Invalid shuffle_over_epochs: {self.shuffle_over_epochs}")

    def _filter_files(self, fname: str, shard_name: str) -> bool:
        """Filter function for webdataset for excluding files from the shards."""
        if shard_name in self.exclude:
            return False

        # Get base_name and extension if available
        m = split_name_re.match(fname)
        if not m:
            return False
        base_name, ext = m.groups()

        if f"{shard_name}/{base_name}" in self.exclude:
            return False
        if self.part_filter is not None and not self.part_filter(ext):
            return False
        return True

    def _tarfile_sample_iter(
        self,
        tar_file: tarfile.TarFile,
        shard_info: ShardInfo,
        absolute_tar_begin_byte_offset: int,
    ) -> Generator[Tuple[FilteredSample, int], None, None]:
        group: Optional[FilteredSample] = None
        last_group_key: Optional[str] = None
        key: str = f"{shard_info.name}/UNKNOWN"

        for tarinfo in tar_file:
            try:
                fname = tarinfo.name
                if not tarinfo.isreg():
                    continue
                if fname is None:
                    continue
                if skip_meta_re.match(fname):
                    continue

                # Get base_name and extension if available
                m = split_name_re.match(fname)
                if not m:
                    continue
                base_name, ext = m.groups()

                key = f"{shard_info.name}/{base_name}"

                if last_group_key != base_name:
                    if group is not None:
                        yield group, tarinfo.offset

                    group = dict(
                        __key__=key,
                        __shard__=shard_info.name,
                        __restore_key__=(
                            "Webdataset",
                            shard_info.name,
                            tarinfo.offset + absolute_tar_begin_byte_offset,
                        ),
                    )
                    last_group_key = base_name

                if self.part_filter is None or self.part_filter(ext):
                    group[ext] = tar_file.extractfile(tarinfo).read()
            except SYSTEM_EXCEPTIONS:
                raise FatalSampleError.from_sample_key(key)
            except Exception as e:
                self.handler(e, key)

        if group is not None:
            # shard_state.byte_offset = (absolute_tar_begin_offset - shard_info.byte_offset)
            yield group, shard_info.byte_size - (
                absolute_tar_begin_byte_offset - shard_info.byte_offset
            )
        else:
            return

    def _shard_iter(self, shard_state: ShardState) -> Generator[FilteredSample, None, None]:
        """Iterates the samples in a shard (potentially resuming from a saved state)."""
        # print(
        #     f"Shard iter for shard {shard_state.shard.name} [{shard_state.shard.offset} +{shard_state.byte_offset}b, {shard_state.shard.offset + shard_state.shard.count} -{shard_state.byte_offset}b) starting"
        # )
        if self.worker_config.should_log(level=2):
            self.worker_config.worker_log(
                {
                    "t": "WebdatasetSampleLoaderDataset._shard_iter",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "shard": {
                        "name": shard_state.shard.name,
                        "path": str(shard_state.shard.path),
                        "offset": shard_state.shard.offset,
                        "count": shard_state.shard.count,
                    },
                    "offset": shard_state.offset,
                }
            )

        shard = shard_state.shard
        if shard.byte_offset is None:
            shard.byte_offset = get_itar_byte_offset(shard.path, shard.offset)
        if shard.byte_size is None:
            shard.byte_size = (
                get_itar_byte_offset(shard.path, shard.offset + shard.count) - shard.byte_offset
            )

        if shard_state.byte_offset == shard.byte_size:
            # Empty shard, return immediately (cannot be handled by the open_itar function)
            return

        # If the shard is not empty, the absolute byte offset must be smaller than the shard size
        assert shard_state.byte_offset < shard.byte_size

        # Given the shard offset (e.g. sub-shard) and the relative byte offset from the stored state, compute the absolute byte offset
        absolute_byte_offset = shard.byte_offset + shard_state.byte_offset
        sub_tar_byte_size = shard.byte_size - shard_state.byte_offset
        orig_shard_state_byte_offset = shard_state.byte_offset

        try:
            # Open the shard and resume where it was last stopped

            with open_itar(
                shard.path,
                byte_offset=absolute_byte_offset,
                byte_size=sub_tar_byte_size,
            ) as it:
                for group, next_sample_offset_in_sub_tar in self._tarfile_sample_iter(
                    it, shard, absolute_byte_offset
                ):
                    if self.worker_config.should_log(level=3):
                        self.worker_config.worker_log(
                            {
                                "t": "WebdatasetSampleLoaderDataset._shard_iter.yield",
                                "r": self.worker_config.rank,
                                "w": self.worker_config.rank_worker_id(),
                                "shard": {
                                    "name": shard_state.shard.name,
                                    "path": str(shard_state.shard.path),
                                    "offset": shard_state.shard.offset,
                                    "count": shard_state.shard.count,
                                },
                                "offset": shard_state.offset,
                                "key": group["__key__"],
                            }
                        )

                    # Next state
                    # NOTE: The next_sample_offset_in_sub_tar (tarinfo.offset) is relative to
                    # absolute_byte_offset, since open_itar crops a part out of the file.
                    # But we want to compute the offset relative to shard.byte_offset
                    shard_state.byte_offset = (
                        next_sample_offset_in_sub_tar + orig_shard_state_byte_offset
                    )
                    assert shard_state.byte_offset <= shard.byte_size
                    # print(f"Yield {group['__key__']} @next: {key} @{shard_state.byte_offset}b")

                    shard_state.offset += 1
                    if group["__key__"] in self.exclude:
                        continue
                    yield group

                # Set to end of subtar
                if shard_state.offset != shard.count or shard_state.byte_offset != shard.byte_size:
                    warnings.warn(
                        f"shard_state.offset({shard_state.offset}) != shard.count({shard.count}) or "
                        f"shard_state.byte_offset({shard_state.byte_offset}) != shard.byte_size({shard.byte_size})"
                        f"; this indicates an internal bug. Shard might not have been interated completely, samples may be missing."
                    )
                # assert shard_state.offset == shard.count
                # assert shard_state.byte_offset == shard.byte_size
        except SYSTEM_EXCEPTIONS:
            raise FatalSampleError.from_sample_key(f"{shard.path}")
        except Exception as e:
            self.handler(e, shard.name)
        # print(f"Shard iter for shard {shard.name} [{shard.offset}, {shard.count}) done")

    def _shards_iter(self, shards: List[ShardInfo]) -> Generator[FilteredSample, None, None]:
        """Iterates the samples in a list of shards, possibly looping over them indefinitely,
        possibly reshuffling the shards every loop, possibly using multiple parallel iterators over
        the shards."""
        worker_idx = self.worker_config.rank_worker_id()
        # Cleanup other states
        for i in range(self.worker_config.num_workers):
            if i != worker_idx:
                self._active_shards_state[i] = [None] * self.parallel_shard_iters
                self._pending_shards[i] = []
        shards_probs = torch.empty(self.parallel_shard_iters, dtype=torch.float32)
        shard_iters: List[Optional[Generator[FilteredSample, None, None]]]
        active_shards: List[Optional[ShardState]]
        while True:
            shards_probs[:] = 0
            shard_iters = []
            if (
                any(s is not None for s in self._active_shards_state[worker_idx])
                or len(self._pending_shards[worker_idx]) > 0
            ):
                shards_order = self._pending_shards[worker_idx]

                # Restore the state
                active_shards = self._active_shards_state[worker_idx]
                assert len(active_shards) == self.parallel_shard_iters
                for shard_state in active_shards:
                    if shard_state is None:
                        shard_iters.append(None)
                    else:
                        shards_probs[len(shard_iters)] = shard_state.shard.count
                        shard_iters.append(self._shard_iter(shard_state))

                if self.worker_config.should_log(level=1):
                    self.worker_config.worker_log(
                        {
                            "t": "WebdatasetSampleLoaderDataset._shards_iter.resume_epoch",
                            "r": self.worker_config.rank,
                            "w": self.worker_config.rank_worker_id(),
                            "shards": [
                                {
                                    "name": shard.name,
                                    "path": str(shard.path),
                                    "offset": shard.offset,
                                    "count": shard.count,
                                }
                                for shard in shards_order
                            ],
                            "active_shards": [
                                (
                                    None
                                    if shard_state is None
                                    else {
                                        "shard": {
                                            "name": shard_state.shard.name,
                                            "path": str(shard_state.shard.path),
                                            "offset": shard_state.shard.offset,
                                            "count": shard_state.shard.count,
                                        },
                                        "offset": shard_state.offset,
                                    }
                                )
                                for shard_state in active_shards
                            ],
                            "count": self._sample_count[worker_idx],
                            "epoch": self._epoch_count[worker_idx],
                            "epoch_count": self._epoch_sample_count[worker_idx],
                            "probs": shards_probs.tolist(),
                        }
                    )

            else:
                # Weight the shards by their size to get a more even distribution of samples
                shards_order = self._shards_once(shards)
                shards_order.reverse()

                if self.worker_config.should_log(level=1):
                    self.worker_config.worker_log(
                        {
                            "t": "WebdatasetSampleLoaderDataset._shards_iter.next_epoch",
                            "r": self.worker_config.rank,
                            "w": self.worker_config.rank_worker_id(),
                            "shards": [
                                {
                                    "name": shard.name,
                                    "path": str(shard.path),
                                    "offset": shard.offset,
                                    "count": shard.count,
                                }
                                for shard in shards_order
                            ],
                            "count": self._sample_count[worker_idx],
                            "epoch": self._epoch_count[worker_idx],
                            "epoch_count": self._epoch_sample_count[worker_idx],
                            "probs": shards_probs.tolist(),
                            "shuffle_over_epochs": self.shuffle_over_epochs,
                        }
                    )

                # List of shard iterators, always of length `parallel_shard_iters`. May contain `None`.
                active_shards = []
                # Fill up the shard iterators
                while len(shards_order) > 0 and len(shard_iters) < self.parallel_shard_iters:
                    shard = shards_order.pop()
                    shard_state = ShardState(shard=shard, byte_offset=0, offset=0)
                    shards_probs[len(shard_iters)] = shard.count
                    shard_iters.append(self._shard_iter(shard_state))
                    active_shards.append(shard_state)
                # Fill up the shard iterators with None
                for _ in range(len(shard_iters), self.parallel_shard_iters):
                    shard_iters.append(None)
                    active_shards.append(None)

                self._active_shards_state[worker_idx] = active_shards
                self._pending_shards[worker_idx] = shards_order

            # print(
            #     f"Next shard iters generated for {self.worker_config.rank}:{self.worker_config.rank_worker_id()}: probs={shards_probs}"
            # )

            # Iterate over the shard iterators
            while True:
                if torch.count_nonzero(shards_probs).item() == 0:
                    # There is no iterator left
                    break
                if self.shuffle_over_epochs is None:
                    # No shuffling, deterministic order, always the same
                    assert self.parallel_shard_iters == 1
                    shard_iter = shard_iters[0]
                else:
                    # Take a random shard iterator
                    shard_iter = self._worker_rng.choice(shard_iters, probs=shards_probs)
                try:
                    sample: FilteredSample = next(shard_iter)
                except StopIteration:
                    # Iterator exhausted -> take next / remove from list
                    rm_idx = shard_iters.index(shard_iter)
                    if len(shards_order) > 0 or self.shuffle_over_epochs == -1:
                        if len(shards_order) > 0:
                            # Take the next shard (without replacement)
                            shard = shards_order.pop()
                        else:
                            # Randomly select a new shard directly (with replacement)
                            shard = self.shards[worker_idx][
                                self._worker_rng.randbelow(len(self.shards[worker_idx]))
                            ]
                        shard_state = ShardState(shard=shard, byte_offset=0, offset=0)
                        shard_iters[rm_idx] = self._shard_iter(shard_state)
                        shards_probs[rm_idx] = shard.count
                        active_shards[rm_idx] = shard_state
                        # print(
                        #     f"Shard iter for {self.worker_config.rank}:{self.worker_config.rank_worker_id()} exhausted, taking next shard {shard.name} [{shard.offset}, {shard.offset + shard.count}), {len(shards_order)} shards left, probs={shards_probs}"
                        # )
                    else:
                        shard_iters[rm_idx] = None
                        shards_probs[rm_idx] = 0
                        active_shards[rm_idx] = None
                        # print(
                        #     f"Shard iter for {self.worker_config.rank}:{self.worker_config.rank_worker_id()} exhausted, no next shards, probs={shards_probs}"
                        # )
                    if self.worker_config.should_log(level=2):
                        self.worker_config.worker_log(
                            {
                                "t": "WebdatasetSampleLoaderDataset._shards_iter.exhausted",
                                "r": self.worker_config.rank,
                                "w": self.worker_config.rank_worker_id(),
                                "remaining": len(shards_order),
                                "count": self._sample_count[worker_idx],
                                "epoch": self._epoch_count[worker_idx],
                                "epoch_count": self._epoch_sample_count[worker_idx],
                                "probs": shards_probs.tolist(),
                            }
                        )
                else:
                    self._sample_count[worker_idx] += 1
                    self._epoch_sample_count[worker_idx] += 1
                    if self.worker_config.should_log(level=1):
                        self.worker_config.worker_log(
                            {
                                "t": "WebdatasetSampleLoaderDataset._shards_iter.yield",
                                "r": self.worker_config.rank,
                                "w": self.worker_config.rank_worker_id(),
                                "key": sample["__key__"],
                                "shard": sample["__shard__"],
                                "count": self._sample_count[worker_idx],
                                "epoch": self._epoch_count[worker_idx],
                                "epoch_count": self._epoch_sample_count[worker_idx],
                            }
                        )
                    yield sample
            if self.worker_config.should_log(level=2):
                self.worker_config.worker_log(
                    {
                        "t": "WebdatasetSampleLoaderDataset._shards_iter.all_exhausted",
                        "r": self.worker_config.rank,
                        "w": self.worker_config.rank_worker_id(),
                        "count": self._sample_count[worker_idx],
                        "epoch": self._epoch_count[worker_idx],
                        "epoch_count": self._epoch_sample_count[worker_idx],
                    }
                )

            self._epoch_count[worker_idx] += 1
            self._epoch_sample_count[worker_idx] = 0
            # print(
            #     f"Shard iters exhausted for {self.worker_config.rank}:{self.worker_config.rank_worker_id()} after {cnt} samples"
            # )
            if not self.loop:
                break

    def __len__(self) -> int:
        return sum(shard.count for worker_shards in self.shards for shard in worker_shards)

    def worker_has_samples(self) -> bool:
        self.worker_config.assert_worker()
        worker_shards = self.shards[self.worker_config.rank_worker_id()]
        return any(shard.count > 0 for shard in worker_shards)

    def __iter__(self) -> Iterator[FilteredSample]:
        self.worker_config.assert_worker()
        worker_shards = self.shards[self.worker_config.rank_worker_id()]

        if self.worker_config.should_log(level=1):
            self.worker_config.worker_log(
                {
                    "t": "WebdatasetSampleLoaderDataset.__iter__",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "shard_range": [
                        f"{shard.name}[{shard.offset}, {shard.offset+shard.count})"
                        for shard in worker_shards
                    ],
                    "parallel_shard_iters": self.parallel_shard_iters,
                    "shuffle_over_epochs": self.shuffle_over_epochs,
                    "loop": self.loop,
                }
            )

        if len(worker_shards) == 0:
            return

        yield from self._shards_iter(worker_shards)

    def can_restore_sample(self) -> bool:
        return True
    
    def assert_can_restore(self) -> None:
        pass

    def restore_sample(self, key: Tuple[Union[str, int, tuple], ...]) -> FilteredSample:
        id, shard_name, tar_byte_offset = key
        assert id == "Webdataset"
        shard_path = self.shard_path_map[shard_name]

        sample_shard_info = ShardInfo(
            name=shard_name,
            path=shard_path,
            offset=0,
            count=1,
            byte_offset=tar_byte_offset,
            # Just a dummy value
            byte_size=0,
        )

        # Open the shard and extract the sample
        with open_itar(shard_path, byte_offset=tar_byte_offset) as tar_file:
            gen = self._tarfile_sample_iter(tar_file, sample_shard_info, tar_byte_offset)
            sample, _ = next(gen)
            gen.close()
            return sample

    def save_state(self) -> SampleLoaderState:
        self.worker_config.assert_worker()
        worker_idx = self.worker_config.rank_worker_id()
        assert len(self._active_shards_state[worker_idx]) == self.parallel_shard_iters
        if self.worker_config.should_log(level=3):
            self.worker_config.worker_log(
                {
                    "t": "WebdatasetSampleLoaderDataset.save_state",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "count": self._sample_count[worker_idx],
                    "epoch": self._epoch_count[worker_idx],
                    "epoch_count": self._epoch_sample_count[worker_idx],
                    "pending_shards": [
                        {
                            "name": shard.name,
                            "path": str(shard.path),
                            "offset": shard.offset,
                            "count": shard.count,
                        }
                        for shard in self._pending_shards[worker_idx]
                    ],
                    "active_shards": [
                        (
                            None
                            if shard_state is None
                            else {
                                "shard": {
                                    "name": shard_state.shard.name,
                                    "path": str(shard_state.shard.path),
                                    "offset": shard_state.shard.offset,
                                    "count": shard_state.shard.count,
                                },
                                "offset": shard_state.offset,
                            }
                        )
                        for shard_state in self._active_shards_state[worker_idx]
                    ],
                }
            )
        return SampleLoaderState(
            rng=self._worker_rng.save_state(),
            pending_shards=list(self._pending_shards[worker_idx]),
            active_shards=[
                None if active_shard is None else dataclasses.replace(active_shard)
                for active_shard in self._active_shards_state[worker_idx]
            ],
            sample_count=self._sample_count[worker_idx],
            epoch_count=self._epoch_count[worker_idx],
            epoch_sample_count=self._epoch_sample_count[worker_idx],
        )

    def merge_states(self, states: List[SampleLoaderState]) -> SampleLoaderMergedState:
        assert all(s is None or isinstance(s, SampleLoaderState) for s in states)
        assert len(states) == len(self.shards)
        return SampleLoaderMergedState(
            rng=self._worker_rng.merge_states([None if s is None else s.rng for s in states]),
            pending_shards=[[] if s is None else s.pending_shards for s in states],
            active_shards=[
                [None] * self.parallel_shard_iters if s is None else s.active_shards for s in states
            ],
            sample_count=[0 if s is None else s.sample_count for s in states],
            epoch_count=[0 if s is None else s.epoch_count for s in states],
            epoch_sample_count=[0 if s is None else s.epoch_sample_count for s in states],
        )

    def _restore_find_shard(
        self, shard_data: ShardInfo, shards_by_key: Dict[Tuple[str, int], ShardInfo]
    ) -> ShardInfo:
        shard = shards_by_key[(shard_data.name, shard_data.offset)]
        if shard != shard_data:
            raise ValueError(
                f"Shard {shard_data!r} not found in {self.shards!r}, states differ, not recoverable"
            )
        # Copy over the byte size and offset. Saves some loading time,
        # especially for restoring lots of random samples :)
        if shard_data.byte_offset is not None and shard.byte_offset is None:
            shard.byte_offset = shard_data.byte_offset
        if shard_data.byte_size is not None and shard.byte_size is None:
            shard.byte_size = shard_data.byte_size
        return shard

    def restore_state(self, state: Optional[SampleLoaderMergedState]) -> None:
        if self.worker_config.should_log(level=3):
            self.worker_config.worker_log(
                {
                    "t": "WebdatasetSampleLoaderDataset.restore_state",
                    "r": self.worker_config.rank,
                    "w": self.worker_config.rank_worker_id(),
                    "state": str(state),
                }
            )
        # print(f"Restore state {state}")
        if state is None:
            # Restore initial state
            self._worker_rng.restore_state(None)
            self._pending_shards = [[] for _ in range(len(self.shards))]
            self._active_shards_state = [
                [None] * self.parallel_shard_iters for _ in range(len(self.shards))
            ]
            self._sample_count = [0] * len(self.shards)
            self._epoch_count = [0] * len(self.shards)
            self._epoch_sample_count = [0] * len(self.shards)
        else:
            assert isinstance(state, SampleLoaderMergedState)
            self._worker_rng.restore_state(state.rng)
            # Restore state
            assert len(state.pending_shards) == len(self.shards)
            assert len(state.active_shards) == len(self.shards)
            self._pending_shards = [
                [self._restore_find_shard(shard, shards_by_key) for shard in pending_shards]
                for pending_shards, shards_by_key in zip(state.pending_shards, self.shards_by_key)
            ]
            self._active_shards_state = [
                [
                    (
                        None
                        if shard_state is None
                        else ShardState(
                            shard=self._restore_find_shard(shard_state.shard, shards_by_key),
                            offset=shard_state.offset,
                            byte_offset=shard_state.byte_offset,
                        )
                    )
                    for shard_state in active_shard_states
                ]
                for active_shard_states, shards_by_key in zip(
                    state.active_shards, self.shards_by_key
                )
            ]
            self._sample_count = state.sample_count
            self._epoch_count = state.epoch_count
            self._epoch_sample_count = state.epoch_sample_count

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "shards": [
                [
                    {
                        "name": shard.name,
                        "path": str(shard.path),
                        "offset": shard.offset,
                        "count": shard.count,
                        "byte_offset": shard.byte_offset,
                        "byte_size": shard.byte_size,
                    }
                    for shard in shards
                ]
                for shards in self.shards
            ],
            "worker_config": self.worker_config.config(),
            "exclude": list(self.exclude),
            "loop": self.loop,
            "shuffle_over_epochs": self.shuffle_over_epochs,
            "parallel_shard_iters": self.parallel_shard_iters,
        }

    def __str__(self):
        return f"WebdatasetSampleLoaderDataset(shards={self.shards}, loop={self.loop}, shuffle_over_epochs={self.shuffle_over_epochs}, parallel_shard_iters={self.parallel_shard_iters})"
