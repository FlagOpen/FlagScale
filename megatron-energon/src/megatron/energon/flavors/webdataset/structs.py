# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TypedDict

from megatron.energon.epathlib import EPath


@dataclass
class WebdatasetInfo:
    # Maps shard name to number of samples in that shard
    shard_counts: Dict[str, int]


@dataclass
class WebdatasetSplits:
    # Maps split part to list of shard names
    split_parts: Dict[str, List[str]]
    # Set of "<shard name>" or "<shard name>/<sample index>" to exclude
    exclude: List[str] = field(default_factory=list)


@dataclass
class ShardInfo:
    #: Name of the shard file (relative path from the nvinfo dir)
    name: str
    #: The path to the shard file
    path: EPath
    #: The offset of the first sample (counted in samples) where this sub shard starts
    offset: int
    #: The number of samples in this sub shard
    count: int
    #: The absolute file offset in bytes where this sub shard starts, None if not computed yet
    byte_offset: Optional[int] = field(default=None, compare=False)
    #: The length of the sub shard in bytes within the shard file, None if not computed yet
    byte_size: Optional[int] = field(default=None, compare=False)


class FilteredSample(TypedDict):
    #: The key of the sample within the tar file.
    #: If the tar file contains files 12.jpg and 12.txt,
    #: those two files make one sample with the key "12"
    __key__: str
    #: The base name of the shard file e.g. "shard_000"
    __shard__: str
    #: Globally unique key to restore a sample from disk.
    #: For example `("shard_000", 123)` would restore
    #: the sample from "shard_000.tar" at byte offset 123.
    __restore_key__: Tuple[str, int]


def reraise_exception(exc: Exception, key: Optional[str]) -> None:
    if key:
        raise Exception(f"For sample {key!r}") from exc
    else:
        raise
