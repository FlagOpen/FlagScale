# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
from pathlib import Path as PathlibPath
from pathlib import PurePosixPath
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import fsspec
import s3fs
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

from megatron.energon.epathlib.rclone_config import ConfigEntry, read_rclone_config


class EPath:
    """EPath work much like pathlib.Path but does support local as well as remote file systems.
    To initialize a remote path, prepend the path with "rclone://". For example:

        EPath("rclone://myrclonename/myfile.txt")

    You will need to have your rclone configuration set up to access the remote file system.
    Currently only S3-based rclone remotes are supported.
    """

    internal_path: PurePosixPath  # The path without the protocol. Can also be in S3 for example
    fs: AbstractFileSystem
    protocol: str
    num_fs_path_parts: (
        int  # Number of parts in the internal_path that are part of the file system specification
    )
    s3_args: Optional[Dict[str, Any]]

    def __init__(
        self,
        initial_path: Union[str, "EPath", PathlibPath],
        config_override: Optional[Dict[str, ConfigEntry]] = None,
    ) -> None:
        self.s3_args = None

        if isinstance(initial_path, EPath):
            self.internal_path = initial_path.internal_path
            self.fs = initial_path.fs
            self.protocol = initial_path.protocol
            self.num_fs_path_parts = initial_path.num_fs_path_parts
            self.s3_args = initial_path.s3_args
        elif isinstance(initial_path, PathlibPath):
            self.internal_path = PurePosixPath(initial_path.absolute())
            self.fs = LocalFileSystem()
            self.protocol = "local"
            self.num_fs_path_parts = 0
        elif isinstance(initial_path, str):
            protocol, path = self._split_protocol(initial_path)
            if protocol is None:
                self.internal_path = PurePosixPath(path)
                self.fs = LocalFileSystem()
                self.protocol = "local"
                self.num_fs_path_parts = 0
            elif protocol == "rclone":
                if not path.startswith("/"):
                    # rclone paths are treated as absolute
                    # The first part of the path will be the rclone remote name
                    path = "/" + path

                self.internal_path = self._resolve(path)

                remote_name = self.internal_path.parts[1]
                self.fs, self.s3_args = self.create_s3fs_from_rclone_remote(
                    remote_name, config_override=config_override
                )
                self.protocol = "rclone"
                self.num_fs_path_parts = 2  # Root and remote name
            else:
                raise ValueError(f"Unknown protocol: {protocol}")

    @staticmethod
    def _resolve(path: Union[str, PurePosixPath]) -> PurePosixPath:
        """Resolve a path, removing .. and . components."""
        if isinstance(path, str):
            path = PurePosixPath(path)
        parts = path.parts
        if parts[0] != "/":
            raise ValueError("Only absolute paths are supported")
        if ".." in parts or "." in parts:
            new_parts = []
            for part in parts[1:]:
                if part == "..":
                    if len(new_parts) == 0:
                        raise ValueError(f"Path above root: {path}")
                    new_parts.pop()
                elif path == ".":
                    pass
                else:
                    new_parts.append(part)
            path = PurePosixPath("/", *new_parts)
        return path

    @staticmethod
    def _split_protocol(path):
        regex = re.compile(r"^(?P<protocol>[a-z]+)://(?P<path>.+)$")
        m = regex.match(path)
        if m is None:
            return None, path
        return m.group("protocol"), m.group("path")

    @staticmethod
    def create_s3fs_from_rclone_remote(
        remote_name: str, config_override: Optional[Dict[str, ConfigEntry]] = None
    ) -> Tuple[s3fs.S3FileSystem, Dict[str, Any]]:
        """Given an rclone remote name (as in rclone.conf), create an S3FileSystem."""

        # Now read the rclone config file to get the endpoint URL and credentials
        if config_override is not None:
            rclone_config = config_override
        else:
            rclone_config = read_rclone_config()

        if remote_name not in rclone_config:
            raise ValueError(f"Unknown rclone remote: {remote_name}")

        rclone_entry = rclone_config[remote_name]

        if rclone_entry.type.lower() != "s3":
            raise NotImplementedError(
                f"Only S3 rclone remotes are supported. Remote {remote_name} is of type {rclone_entry.type}"
            )

        # If the endpoint does not specify a protocol, assume https://
        if rclone_entry.endpoint and not rclone_entry.endpoint.startswith("http"):
            s3fs_endpoint = "https://" + rclone_entry.endpoint
        else:
            s3fs_endpoint = rclone_entry.endpoint

        s3_args = dict(
            anon=False,
            key=rclone_entry.access_key_id,
            secret=rclone_entry.secret_access_key,
            endpoint_url=s3fs_endpoint,
            client_kwargs={
                "verify": False,
                "region_name": rclone_entry.region,
            },
        )

        return s3fs.S3FileSystem(**s3_args), s3_args  # type: ignore

    @staticmethod
    def prepare_forked_process():
        """Clear the cache of instances created by EPath and the async stuff.
        This is needed to avoid issues when forking a process that uses EPath.

        This static method needs to be called once in the child process after forking.
        It is independent of any EPath instances. For those, see the fork_guard() below.
        """

        fsspec.asyn.iothread[0] = None  # type: ignore
        fsspec.asyn.loop[0] = None  # type: ignore
        s3fs.S3FileSystem.clear_instance_cache()

    def fork_guard(self):
        """Check if this EPath instance is a forked copy with S3FS and re-create S3FS if needed.

        This method should be called before any operation that potentially uses the S3FS instance.
        """

        if isinstance(self.fs, s3fs.S3FileSystem) and self.fs._pid != os.getpid():
            assert self.s3_args is not None
            self.fs = s3fs.S3FileSystem(**self.s3_args)

    @property
    def _internal_fs_path(self) -> str:
        """Return the part of path the path that specifies FS properties like rclone rmeote name."""
        return str(PurePosixPath("/", *self.internal_path.parts[: self.num_fs_path_parts]))

    @property
    def _internal_nonfs_path(self) -> str:
        """Return the path as used inside the file system, without the protocol and fs part."""
        return str(PurePosixPath("/", *self.internal_path.parts[self.num_fs_path_parts :]))

    def open(self, mode="r", block_size=None):
        self.fork_guard()
        assert self.is_absolute()
        return self.fs.open(self._internal_nonfs_path, mode, block_size=block_size)

    def read_text(self):
        with self.open() as f:
            return f.read()

    @property
    def name(self):
        return self.internal_path.name

    @property
    def parent(self):
        new_path = EPath(self)
        new_path.internal_path = self.internal_path.parent
        return new_path

    @property
    def url(self):
        assert self.is_absolute(), "Can only call url on absolute EPath"
        if self.protocol == "rclone":
            int_path_str = str(self.internal_path)
            if int_path_str.startswith("/"):
                # Strip leading / for display purposes
                int_path_str = int_path_str[1:]
            return f"{self.protocol}://{int_path_str}"
        else:
            return str(self.internal_path)

    @property
    def relpath(self):
        assert not self.is_absolute(), "Can only call relpath on relative EPath"
        return str(self.internal_path)

    def is_absolute(self):
        return self.internal_path.is_absolute()

    def absolute(self) -> "EPath":
        if self.protocol == "local":
            if self.is_absolute():
                return self
            return EPath(PathlibPath(self.internal_path).absolute())
        else:
            assert self.is_absolute(), "Remote paths are always absolute"
            return self

    def is_dir(self):
        self.fork_guard()
        assert self.is_absolute()
        return self.fs.isdir(self._internal_nonfs_path)

    def is_file(self):
        self.fork_guard()
        assert self.is_absolute()
        return self.fs.isfile(self._internal_nonfs_path)

    def mkdir(self, exist_ok: bool = True, parents: bool = False):
        self.fork_guard()
        assert self.is_absolute()
        if parents:
            return self.fs.makedirs(self._internal_nonfs_path, exist_ok=exist_ok)
        else:
            try:
                return self.fs.mkdir(self._internal_nonfs_path)
            except FileExistsError:
                if exist_ok:
                    pass
                else:
                    raise

    def glob(self, pattern) -> Generator["EPath", None, None]:
        self.fork_guard()
        assert self.is_absolute()

        search_path_pattern = (self / pattern)._internal_nonfs_path

        if self.protocol != "local" and search_path_pattern.startswith("/"):
            # For some reason s3fs glob does not like leading /
            search_path_pattern = search_path_pattern[1:]

        for path in self.fs.glob(search_path_pattern):
            assert isinstance(path, str)

            new_path = EPath(self)  # Copy

            if self.protocol == "local":
                assert path.startswith("/"), "Local FS glob should return absolute paths"
                new_path.internal_path = self._resolve(path)
            else:
                new_path.internal_path = self._resolve(self._internal_fs_path / PurePosixPath(path))

            yield new_path

    def size(self):
        self.fork_guard()
        assert self.is_absolute()
        return self.fs.size(self._internal_nonfs_path)

    def with_suffix(self, suffix):
        new_path = EPath(self)
        new_path.internal_path = self.internal_path.with_suffix(suffix)
        return new_path

    def move(self, target: "EPath"):
        self.fork_guard()
        assert self.is_absolute()
        assert target.is_absolute()

        assert self.fs == target.fs, "Can only move within same FS"
        assert self.protocol == target.protocol, "Can only move within same FS"
        assert self._internal_fs_path == target._internal_fs_path, "Can only move within same FS"

        self.fs.mv(self._internal_nonfs_path, target._internal_nonfs_path)

    def unlink(self):
        self.fork_guard()
        assert self.is_absolute()
        return self.fs.rm(self._internal_nonfs_path)

    def relative_to(self, other: "EPath") -> str:
        self.fork_guard()
        assert self.is_absolute()
        assert other.is_absolute()

        assert self.fs == other.fs, "Can only use relative_to within same FS"
        assert self.protocol == other.protocol, "Can only use relative_to within same FS"
        assert (
            self._internal_fs_path == other._internal_fs_path
        ), "Can only use relative_to within same FS"

        return str(self.internal_path.relative_to(other.internal_path))

    def __truediv__(self, other: Union[str, "EPath"]):
        assert self.is_absolute()

        other = EPath(other) if isinstance(other, str) else other
        if other.is_absolute():
            # If the second path is absolute, just return it as is
            return other

        new_path = EPath(self)
        new_path.internal_path = self._resolve(
            self.internal_path / (other if isinstance(other, str) else other.internal_path)
        )
        return new_path

    def __rtruediv__(self, other: "EPath"):
        assert other.is_absolute()
        # assert not self.is_absolute()

        if self.is_absolute():
            # If second path (self) is absolute, just return it as is
            return self

        new_path = EPath(self)
        new_path.internal_path = self._resolve(other.internal_path / self.internal_path)
        return new_path

    def __lt__(self, other: "EPath"):
        self.fork_guard()
        assert self.fs == other.fs, "Cannot compare paths from different file systems"
        assert self.protocol == other.protocol, "Cannot compare paths from different file systems"
        assert (
            self._internal_fs_path == other._internal_fs_path
        ), "Cannot compare paths from different file systems"

        return self.internal_path < other.internal_path

    def __str__(self):
        if self.is_absolute():
            return str(self.url)
        else:
            return str(self.relpath)

    def __repr__(self) -> str:
        return f"EPath({str(self)!r})"

    def __hash__(self):
        return hash((self.internal_path, self.protocol))

    def __eq__(self, other):
        return self.internal_path == other.internal_path and self.protocol == other.protocol
