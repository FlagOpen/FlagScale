# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
import logging
import multiprocessing as mp
import random
import re
import tarfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Generator,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import braceexpand
import torch
import webdataset
import yaml

from megatron.energon.dataset_config import MAIN_FOLDER_NAME
from megatron.energon.epathlib import EPath
from megatron.energon.errors import SYSTEM_EXCEPTIONS, FatalSampleError
from megatron.energon.flavors.base_dataset import (
    BaseCoreDataset,
    MergedState,
    Sample,
    SavableDataset,
    State,
)
from megatron.energon.flavors.webdataset.config import skip_meta_re, split_name_re
from megatron.energon.flavors.webdataset.field_access import field_access, split_field_access
from megatron.energon.flavors.webdataset.itar import TarIndexWriter
from megatron.energon.flavors.webdataset.sample_loader import WebdatasetSampleLoaderDataset
from megatron.energon.flavors.webdataset.structs import (
    FilteredSample,
    ShardInfo,
    WebdatasetInfo,
    WebdatasetSplits,
    reraise_exception,
)
from megatron.energon.module_loader import ModuleLoader
from megatron.energon.typed_converter import raw_to_typed, to_json_object
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.iter_map_dataset import IterMapDataset
from megatron.energon.wrappers.map_dataset import MapDataset

T_sample = TypeVar("T_sample", covariant=True)
T = TypeVar("T", covariant=True)

logger = logging.getLogger(__name__)

ImageDecoder = Literal[
    "l8",
    "rgb8",
    "rgba8",
    "l",
    "rgb",
    "rgba",
    "torchl8",
    "torchrgb8",
    "torchrgba8",
    "torchl",
    "torchrgb",
    "torch",
    "torchrgba",
    "pill",
    "pil",
    "pilrgb",
    "pilrgba",
]


@dataclasses.dataclass
class VideoData:
    #: The input video tensor in the shape (frames, channel, h, w)
    frames: torch.Tensor
    #: The input audio frames in the shape (number of channels, number of points)
    aframes: torch.Tensor
    #: Metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    info: Dict[str, Union[bool, int, float, str]]


@dataclass
class WebdatasetState(State):
    dataset_state: State


@dataclass
class WebdatasetMergedState(MergedState):
    dataset_state: MergedState


class BaseWebdataset(BaseCoreDataset[T_sample], Generic[T_sample], ABC):
    """
    Base class for all webdataset loaders. Applies proper sharding across workers.
    """

    path: EPath
    training: bool
    worker_config: WorkerConfig
    info: WebdatasetInfo
    split_config: WebdatasetSplits
    extension_fn: Optional[Callable[[str], bool]]

    shards: List[ShardInfo]
    dataset: SavableDataset[T_sample]
    rank_shards: List[List[ShardInfo]]

    class EmptyDatasetError(Exception):
        """Raised when a dataset is empty."""

    def __init__(
        self,
        path: EPath,
        *,
        split_part: str,
        training: bool,
        worker_config: WorkerConfig,
        shuffle_over_epochs: int = 1,
        parallel_shard_iters: Optional[int] = None,
        max_samples_per_sequence: Optional[int] = None,
        info_config: str = ".info.yaml",
        split_config: str = "split.yaml",
        part_filter: Optional[Callable[[str], bool]] = None,
        handler: Callable[[Exception, Optional[str]], None] = reraise_exception,
    ):
        """
        Constructs the webdataset loader.

        Args:
            path: Root path to the dataset for relative path resolution.
            split_part: Which part to load (e.g. 'train', 'val', 'test').
            training: If true, apply shuffling and loop the dataset.
            worker_config: Configuration for the workers.
            shuffle_over_epochs: Only effective if training=True.
                How many epochs to shuffle over if training.
                If = 1, every sample is seen exactly once per epoch.
                If > 1, samples (or rather shard slices) are shuffled within this number of epochs
                (i.e. randomly selected without replacement).
                If -1, the shards are effectively shuffle over infinite epochs (i.e. shard slices
                are drawn with replacement).
            parallel_shard_iters: Number of parallel opened shards per worker, shuffling between.
            max_samples_per_sequence: Maximum number of samples per sequence (=how many samples
                    will be sequentially iterated).
            info_config: Config file to use for sample metadata.
            split_config: Config file to use for shard split definitions.
            part_filter: (internal) Function for filtering tar files by dict keys
            handler: Exception handler. Args: (exception, key).
        """
        assert self.__sample_type__ is not None, f"Class {type(self)} must define __sample_type__"
        self.path = path
        self.training = training
        self.worker_config = worker_config
        self.handler = handler
        self.info = raw_to_typed(
            yaml.safe_load((path / MAIN_FOLDER_NAME / info_config).read_text()), WebdatasetInfo
        )
        self.split_config = raw_to_typed(
            yaml.safe_load((path / MAIN_FOLDER_NAME / split_config).read_text()), WebdatasetSplits
        )
        assert split_part in self.split_config.split_parts, f"Invalid split part: {split_part!r}"
        self.shards = [
            ShardInfo(
                name=name,
                path=path / name,
                offset=0,
                count=self.info.shard_counts[name],
            )
            for name in self.split_config.split_parts[split_part]
            for name in braceexpand.braceexpand(name)
        ]
        if len(self.shards) == 0:
            raise BaseWebdataset.EmptyDatasetError(f"No shards found in split part {split_part!r}")

        exclude = {
            excluded
            for excluded in self.split_config.exclude
            for excluded in braceexpand.braceexpand(excluded)
        }
        total_shard_list = [shard for shard in self.shards if shard.name not in exclude]
        assert len(total_shard_list) > 0, f"No shards left after filtering."

        if parallel_shard_iters is None:
            if training:
                # 16 seems to be a good choice since we don't want too many file handles open
                parallel_shard_iters = 16
            else:
                parallel_shard_iters = 1

        self.rank_shards = self._shard_workers(
            total_shard_list,
            self.worker_config,
            max_samples_per_sequence=max_samples_per_sequence,
        )

        self.rank_total = sum(shard.count for shards in self.rank_shards for shard in shards)
        for idx, shards in enumerate(self.rank_shards):
            shards_text = ", ".join(
                f"{shard.name}[{shard.offset}, {shard.offset+shard.count})" for shard in shards[:3]
            )
            if len(shards) > 6:
                shards_text += f", ...<{len(shards) - 6}>, " + ", ".join(
                    f"{shard.name}[{shard.offset}, {shard.offset+shard.count})"
                    for shard in shards[-3:]
                )
            elif len(shards) > 3:
                shards_text += ", " + ", ".join(
                    f"{shard.name}[{shard.offset}, {shard.offset+shard.count})"
                    for shard in shards[3:]
                )
            print(
                f"rank={self.worker_config.rank}, worker={idx}: shard_range="
                f"[{shards_text}] "
                f"sum(count)={sum(s.count for s in shards)}"
            )

        dataset = WebdatasetSampleLoaderDataset(
            rank_shards=self.rank_shards,
            worker_config=self.worker_config,
            part_filter=part_filter,
            exclude=exclude,
            loop=training,
            shuffle_over_epochs=shuffle_over_epochs if training else None,
            parallel_shard_iters=parallel_shard_iters,
            handler=self.sample_error_handler,
        )
        self.dataset = self._process_samples(dataset)

    def sample_error_handler(self, e: Exception, sample_key: str):
        if isinstance(e, SYSTEM_EXCEPTIONS):
            raise FatalSampleError(f"Error in sample {sample_key!r}: {e}") from e

        self.handler(e, sample_key)

    def error_handler(self, e: Exception, sample: Union[T_sample, List[T_sample], dict]):
        if isinstance(sample, dict):
            key = sample.get("__key__")
        elif isinstance(sample, list):
            if isinstance(sample[0], dict):
                key = ",".join(s.get("__key__") for s in sample)
            elif isinstance(sample[0], Sample):
                key = ",".join(s.__key__ for s in sample)
            else:
                key = None
        elif isinstance(sample, Sample):
            key = sample.__key__
        else:
            key = None
        self.sample_error_handler(e, key)

    @abstractmethod
    def _process_samples(self, dataset: SavableDataset[FilteredSample]) -> SavableDataset[T_sample]:
        """Internally loads the sample."""
        ...

    @staticmethod
    def _split_shard(
        shard: ShardInfo,
        start_offset: int,
        end_offset: int,
        max_samples_per_sequence: Optional[int],
    ) -> List[ShardInfo]:
        if (
            max_samples_per_sequence is not None
            and end_offset - start_offset > max_samples_per_sequence * 1.5
        ):
            # Split the shard into slices of max_samples_per_sequence (more or less)
            slice_count = max(round((end_offset - start_offset) / max_samples_per_sequence), 1)
            samples_per_sequence = (end_offset - start_offset) / slice_count
            # Note this must include the end offset as well, so slice_count + 1 steps (down there,
            # idx+1 is used to access the end offset)
            offsets = [
                start_offset + int(slice * samples_per_sequence) for slice in range(slice_count + 1)
            ]
            return [
                dataclasses.replace(
                    shard,
                    offset=offsets[idx],
                    count=offsets[idx + 1] - offsets[idx],
                    byte_offset=None,
                    byte_size=None,
                )
                for idx in range(slice_count)
            ]
        else:
            return [
                dataclasses.replace(
                    shard,
                    offset=start_offset,
                    count=end_offset - start_offset,
                    byte_offset=None,
                    byte_size=None,
                )
            ]

    @staticmethod
    def _split_shards(
        shards: List[ShardInfo], offsets: List[int], *, max_samples_per_sequence: Optional[int]
    ) -> Generator[List[ShardInfo], None, None]:
        """
        Splits the shards into multiple lists based on the offsets. The first offset is the start
        of the first shard emitted, the last offset is the end of the last shard emitted.
        (i.e. number of shards emitted is `len(offsets) - 1`)

        Args:
            shards: The source shards
            offsets: The offsets to samples to get shards for (must be strictly increasing)
            max_samples_per_sequence: Maximum number of samples per sequence (=how many samples
                  will be sequential).

        Returns:
            A list of shards for each offset pair
        """
        # The start index of the current shard
        cum_count = 0

        # Find shard idx for start
        for start_index, start_shard in enumerate(shards):
            if cum_count + start_shard.count < offsets[0]:
                # The shard is before the offset -> go to next shard
                cum_count += start_shard.count
                continue
            else:
                # The shard contains the offset
                start_offset = offsets[0] - cum_count
                break
        else:
            raise ValueError("Invalid shard distribution")

        for offset in offsets[1:]:
            # Find shard idx for end
            for end_index, end_shard in enumerate(shards[start_index:], start=start_index):
                if cum_count + end_shard.count < offset:
                    # The shard is before the offset -> go to next shard
                    cum_count += end_shard.count
                    continue
                else:
                    # The shard contains the offset
                    end_offset = offset - cum_count
                    break
            else:
                raise ValueError("Invalid shard distribution")
            if start_index == end_index:
                yield BaseWebdataset._split_shard(
                    start_shard,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    max_samples_per_sequence=max_samples_per_sequence,
                )
            else:
                # Middle is the original shards, start and end get an offset/length
                yield (
                    (
                        BaseWebdataset._split_shard(
                            start_shard,
                            start_offset=start_offset,
                            end_offset=start_shard.count,
                            max_samples_per_sequence=max_samples_per_sequence,
                        )
                        if start_shard.count > start_offset
                        else []
                    )
                    + sum(
                        (
                            BaseWebdataset._split_shard(
                                shard,
                                start_offset=shard.offset,
                                end_offset=shard.count,
                                max_samples_per_sequence=max_samples_per_sequence,
                            )
                            for shard in shards[start_index + 1 : end_index]
                        ),
                        start=[],
                    )
                    + BaseWebdataset._split_shard(
                        end_shard,
                        start_offset=end_shard.offset,
                        end_offset=end_offset,
                        max_samples_per_sequence=max_samples_per_sequence,
                    )
                )
            start_index = end_index
            start_shard = end_shard
            start_offset = end_offset

    @classmethod
    def _shard_workers(
        cls,
        shards: List[ShardInfo],
        worker_config: WorkerConfig,
        *,
        max_samples_per_sequence: Optional[int],
    ) -> List[List[ShardInfo]]:
        """
        Creates subshards (ShardInfo) for each worker of the current rank.
        For that, the total number of samples is split into the number of global workers across all
        ranks. Then each worker gets a slice of the global samples.

        Args:
            shards: The shards to split
            worker_config: The config for the current rank and workers

        Returns:
            The shards for the current rank and all workers
        """

        # We split the total number of samples into the number of global workers across all ranks.
        # Note that the global number of workers intentionally stay the same if you
        # divide the number of ranks by N, and multiply the number of workers per rank by N.
        # This allows to reproduce the same global batches with a different number of ranks.

        num_workers = max(1, worker_config.num_workers)

        total_samples = sum(shard.count for shard in shards)
        global_workers = num_workers * worker_config.world_size

        samples_per_worker = total_samples / global_workers

        # Let's look at the workers of the current local rank.
        # Each worker gets a slice of the global samples as follows:
        local_rank_worker_sample_offsets = [
            int((num_workers * worker_config.rank + local_worker_idx) * samples_per_worker)
            for local_worker_idx in range(num_workers + 1)
        ]

        return list(
            # Filter out any empty shards for this worker
            [s for s in shards if s.count > 0]
            for shards in cls._split_shards(
                shards,
                local_rank_worker_sample_offsets,
                max_samples_per_sequence=max_samples_per_sequence,
            )
        )

    def __len__(self):
        # In the training case, the result is an approximation (i.e. number of different samples)
        return self.rank_total

    def __iter__(self) -> Iterator[T_sample]:
        yield from self.dataset

    def worker_has_samples(self) -> bool:
        return self.dataset.worker_has_samples()

    def save_state(self) -> WebdatasetState:
        return WebdatasetState(
            dataset_state=self.dataset.save_state(),
        )

    def merge_states(self, states: List[WebdatasetState]) -> WebdatasetMergedState:
        assert all(s is None or isinstance(s, WebdatasetState) for s in states)
        return WebdatasetMergedState(
            dataset_state=self.dataset.merge_states(
                [None if s is None else s.dataset_state for s in states]
            ),
        )

    def restore_state(self, state: Optional[WebdatasetMergedState]) -> None:
        if state is None:
            self.dataset.restore_state(None)
        else:
            assert isinstance(state, WebdatasetMergedState)
            self.dataset.restore_state(state.dataset_state)

    @staticmethod
    def _preprocess_tar(
        path: Union[str, EPath], parent_path: Union[str, EPath], max_parts: int
    ) -> Tuple[ShardInfo, Set[str]]:
        """Process a single tar file, i.e. read the tarinfos, generate the tar index and return
        stats.

        Args:
            path: Path to the tar file.
            parent_path: Root path of the dataset.
            max_parts: Maximum number of different parts to return

        Returns:
            Tuple of shard info and found keys of the loaded dicts.
        """
        EPath.prepare_forked_process()  # Multiproc with fork

        path = EPath(path)
        shard_info = ShardInfo(name=path.relpath, path=path, offset=0, count=0)

        if not shard_info.path.is_absolute():
            parent_path = EPath(parent_path)
            assert parent_path.is_absolute(), f"Parent path must be absolute: {parent_path}"
            shard_info.path = parent_path / path

        try:
            # Note: Write to .tmp file first, then remove .tmp extension, to make sure only complete
            # files are used.
            with shard_info.path.open("rb") as f:
                with tarfile.open(fileobj=f, mode="r:*") as tar, TarIndexWriter(
                    shard_info.path
                ) as iw:
                    count = 0
                    parts = set()
                    last_base_name = None
                    for member in tar:
                        if not member.isreg():
                            continue
                        if member.name is None:
                            continue
                        if skip_meta_re.match(member.name):
                            continue

                        name_match = split_name_re.match(member.name)
                        if name_match is None:
                            continue

                        base_name = name_match.group(1)
                        if len(parts) < max_parts:
                            parts.add(name_match.group(2))

                        if last_base_name != base_name:
                            iw.append(member.offset)
                            last_base_name = base_name
                            count += 1
                    shard_info.count = count
                    iw.append(tar.offset)
            return shard_info, parts
        except BaseException:
            logger.exception(f"Shard failed to load: {path!r}. Skipping it.")
            return shard_info, set()

    @staticmethod
    def iter_dataset_content(
        path: Union[str, EPath],
        extract_keys: Container[str] = (),
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Yield example dataset content for a few samples.

        Args:
            path: Path to the tar file.
        """
        path = EPath(path)
        with path.open("rb") as f:
            with tarfile.open(fileobj=f, mode="r:*") as tar:
                last_base_name = None
                sample = {}
                for member in tar:
                    if not member.isreg():
                        continue
                    if member.name is None:
                        continue
                    if skip_meta_re.match(member.name):
                        continue

                    name_match = split_name_re.match(member.name)
                    if name_match is None:
                        continue

                    base_name = name_match.group(1)
                    if last_base_name != base_name:
                        if sample:
                            yield sample
                        sample = {}
                        last_base_name = base_name
                    if name_match:
                        if name_match.group(2) in extract_keys:
                            sample[name_match.group(2)] = tar.extractfile(member).read()
                        else:
                            sample[name_match.group(2)] = None
                if sample:
                    yield sample

    @staticmethod
    def prepare_dataset(
        parent_path: Union[Path, EPath],
        paths: List[str],
        *,
        split_parts_ratio: Optional[List[Tuple[str, float]]] = None,
        split_parts_patterns: Optional[List[Tuple[str, str]]] = None,
        info_config: str = ".info.yaml",
        split_config: str = "split.yaml",
        shuffle_seed: Optional[int] = 42,
        progress_fn: Callable[[List, int], Iterable] = (lambda x, l: x),
        workers: int = 32,
        tar_index_only: bool = False,
    ) -> Set[str]:
        """
        Preprocess the shards and write the split config. Preprocessing is done in parallel.
        Counts the number of samples in each shard.

        Args:
            parent_path: Common parent path for the shards
            paths: Paths to the shards
            split_parts_ratio: Names of splits and their ratio (will be normalized)
            split_parts_patterns: Names of splits and their path patterns
            info_config: Filename for the info config (`parent_path / '.nv-meta' / info_config`)
            split_config: Filename for the info config (`parent_path / '.nv-meta' / split_config`)
            shuffle_seed: Seed for shuffling shards before splitting into split_parts. None to
                disable.
            progress_fn: Callback for progress bar
            workers: Number of parallel workers for reading each shard
            tar_index_only: Only create tar-index, then exit

        Returns:
            The set of all parts found in the shards. But at most 50.
        """
        parent_path = EPath(parent_path).absolute()

        found_parts = set()
        paths = [path for path in paths for path in braceexpand.braceexpand(path)]
        shards = []

        assert parent_path.is_absolute(), f"Parent path must be absolute: {parent_path}"

        # use functools partial to pass parent_path to process_tar
        process_tar = functools.partial(
            BaseWebdataset._preprocess_tar,
            parent_path=parent_path.url,  # convert to url string, to avoid EPath in multiprocessing
            max_parts=50,
        )

        with mp.Pool(workers) as pool:
            for shard_info, cur_parts in progress_fn(pool.imap(process_tar, paths), len(paths)):
                if shard_info.count == 0:
                    # This shard failed to load. Skip it.
                    continue
                shards.append(shard_info)
                if len(found_parts) < 50:
                    found_parts.update(cur_parts)

        if tar_index_only:
            return found_parts

        (parent_path / MAIN_FOLDER_NAME).mkdir(exist_ok=True)

        # Save info
        info = WebdatasetInfo(
            shard_counts={shard.name: shard.count for shard in shards},
        )
        with (parent_path / MAIN_FOLDER_NAME / info_config).open("w") as wf:
            yaml.dump(to_json_object(info), wf)

        if split_parts_ratio is not None:
            # Normalize ratio
            total_ratio = sum(split_ratio for _, split_ratio in split_parts_ratio)
            split_parts_ratio = [
                (split_part, split_ratio / total_ratio)
                for split_part, split_ratio in split_parts_ratio
            ]
            # Sample from shards based on the split ratio from split parts
            split_shards = {}
            if shuffle_seed is not None:
                random.Random(shuffle_seed).shuffle(shards)
            split_total = 0
            split_offset = 0
            for split_part, split_ratio in split_parts_ratio:
                split_total += split_ratio
                split_end = int(len(shards) * split_total)
                split_shards[split_part] = [shard.name for shard in shards[split_offset:split_end]]
                split_offset = split_end
        else:
            assert (
                split_parts_patterns is not None
            ), "Require either split_parts_ratio or split_parts_patterns"
            # Sample from shards based on the split patterns from split parts
            split_shards = {}
            for split_part, split_pattern in split_parts_patterns:
                patterns = [
                    re.compile(pattern) for pattern in braceexpand.braceexpand(split_pattern)
                ]
                split_shards[split_part] = [
                    shard.name
                    for shard in shards
                    if any(pattern.match(shard.name) for pattern in patterns)
                ]
        # Save split config
        splits_config = WebdatasetSplits(split_parts=split_shards)
        with (parent_path / MAIN_FOLDER_NAME / split_config).open("w") as wf:
            yaml.dump(to_json_object(splits_config), wf)

        return found_parts

    def can_restore_sample(self) -> bool:
        return True
    
    def assert_can_restore(self):
        pass

    def restore_sample(self, key: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        return self.dataset.restore_sample(key)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "training": self.training,
            "path": str(self.path),
            "worker_config": self.worker_config.config(),
            "dataset": self.dataset.config(),
        }

    def __str__(self):
        return f"{type(self).__name__}(path={self.path}, dataset={self.dataset})"


class DefaultGenericWebdataset(BaseWebdataset[T_sample], Generic[T_sample]):
    """
    Default implementation of Webdataset for generic samples and the generic config interface.
    """

    _sample_loader: Callable[[Dict[str, Any]], Dict[str, Any]]

    def __init__(
        self,
        path: EPath,
        *,
        subflavor: Optional[str] = None,
        subflavors: Optional[Dict[str, Any]] = None,
        field_map: Optional[Dict[str, str]] = None,
        sample_loader: Optional[Union[str, Callable[[dict], dict]]] = None,
        part_filter: Optional[Union[str, List[str], Callable[[str], bool]]] = None,
        **kwargs,
    ):
        assert (field_map is None) != (
            sample_loader is None
        ), "Either field_map or sample_loader must be provided."
        if sample_loader is not None:
            assert (
                part_filter is not None
            ), "part_filter must be provided if sample_loader is provided."
            module_loader = ModuleLoader()
            if isinstance(sample_loader, str):
                sample_loader = module_loader.get_function(
                    sample_loader, "sample_loader", relative_path=path / MAIN_FOLDER_NAME
                )
            else:
                assert callable(sample_loader)
                sample_loader = sample_loader
            if isinstance(part_filter, list):
                parts = set(part_filter)
                part_filter = lambda part: part in parts
            elif isinstance(part_filter, str):
                part_filter = module_loader.get_function(
                    part_filter, "part_filter", relative_path=path / MAIN_FOLDER_NAME
                )
            else:
                assert callable(part_filter)
            self._sample_loader = sample_loader
        else:
            assert field_map is not None
            assert part_filter is None
            # Split field map fields by json[field][field]
            fields = {key: split_field_access(field) for key, field in field_map.items()}
            assert set(field.name for field in dataclasses.fields(self.__sample_type__)).issuperset(
                fields.keys()
            ) and set(
                field.name
                for field in dataclasses.fields(self.__sample_type__)
                if field.default is not dataclasses.MISSING
                and field.default_factory is not dataclasses.MISSING
            ).issubset(
                field_map.keys()
            ), f"field_map does not map to type {self.__sample_type__.__name__} fields"
            self._sample_loader = lambda sample: {
                k: field_access(sample, v) for k, v in fields.items()
            }
            parts = set(access[0] for options in fields.values() for access in options)
            part_filter = lambda part: part in parts
        inner_sample_loader = self._sample_loader
        self._sample_loader = lambda sample: {
            "__key__": sample["__key__"],
            **inner_sample_loader(sample),
            "__restore_key__": sample["__restore_key__"],
            "__subflavor__": self.subflavor,
            "__subflavors__": self.subflavors,
        }
        super().__init__(path, **kwargs, part_filter=part_filter)
        self.subflavor = subflavor
        self.subflavors = subflavors or {}

    def _process_samples(self, dataset: SavableDataset[FilteredSample]) -> SavableDataset[T_sample]:
        return MapDataset(
            dataset,
            self._load_sample,
            error_handler=self.error_handler,
            stateless_map_fn=True,
            worker_config=self.worker_config,
        )

    def _load_sample(self, sample: FilteredSample) -> T_sample:
        return self.__sample_type__(**self._sample_loader(sample))

    def config(self) -> Dict[str, Any]:
        return {
            **super().config(),
            "subflavor": self.subflavor,
            "subflavors": self.subflavors,
            "sample_loader": self._function_config(self._sample_loader),
        }


class DefaultDecoderWebdataset(DefaultGenericWebdataset[T_sample], Generic[T_sample]):
    """
    Basic class for any dataset which contains images and / or videos. Applies default wds loading logic for all
    known extensions.
    """

    #: Image decoding result type
    image_decode: ImageDecoder
    #: If true, ignore errors when decoding.
    ignore_decoder_errors: bool

    def __init__(
        self,
        path: EPath,
        *,
        image_decode: ImageDecoder = "torchrgb",
        ignore_decoder_errors: bool = False,
        **kwargs,
    ):
        """
        Decoder dataset.

        Args:
            path: Path to the dataset (passed to parent)
            image_decode: This defines the decoding results.
            ignore_decoder_errors: If true, ignore errors when decoding.
            **kwargs: Args passed to parent constructor
        """
        self.image_decode = image_decode
        self.ignore_decoder_errors = ignore_decoder_errors
        super().__init__(path, **kwargs)

    def _decode_error_handler(self, exc: Exception) -> bool:
        if self.ignore_decoder_errors:
            return True
        raise exc

    def _video_decoder(self, key, data):
        """Extract the video data from default video extensions."""
        # TODO: This function could be more efficient. It will write the data to `/tmp`,
        # then load it using `torchvision.io.video.read_video` which uses `av.open` from pyav.
        # pyav allows providing a file-like object, but torchvision does not expose that interface.
        # (https://github.com/pytorch/vision/issues/8438)
        video = webdataset.torch_video(key, data)
        if video is not None:
            return VideoData(
                frames=video[0].permute((0, 3, 1, 2)),
                aframes=video[1],
                info=video[2],
            )
        return None

    def _process_samples(self, dataset: SavableDataset[FilteredSample]) -> SavableDataset[T_sample]:

        decoder = webdataset.decode(
            self.image_decode,
            self._video_decoder,
            handler=self._decode_error_handler,
        )
        dataset = IterMapDataset(
            dataset,
            decoder,
            error_handler=self.error_handler,
            stateless_iter_fn=True,
            worker_config=self.worker_config,
        )
        return super()._process_samples(dataset)

    def config(self) -> Dict[str, Any]:
        return {
            **super().config(),
            "image_decode": self.image_decode,
            "ignore_decoder_errors": self.ignore_decoder_errors,
        }
