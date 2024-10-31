# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import struct
import tarfile
from types import TracebackType
from typing import BinaryIO, Optional, Type, Union

from megatron.energon.epathlib import EPath
from megatron.energon.retry_stream import RetryReadStream

ITAR_SUFFIX = ".tar.idx"


class TarIndexReader:
    def __init__(self, tar_path: Union[EPath, str]):
        tar_path = EPath(tar_path)
        self.itar = (tar_path.with_suffix(ITAR_SUFFIX)).open("rb")
        self._length = len(self)

    def __getitem__(self, index: int) -> int:
        if index >= self._length or index < 0:
            raise IndexError(f"Index {index} out of range")

        self.itar.seek(8 * index)
        return struct.unpack("Q", self.itar.read(8))[0]

    def __len__(self) -> int:
        self.itar.seek(0, 2)
        return self.itar.tell() // 8

    def close(self):
        self.itar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TarIndexWriter:
    def __init__(self, tar_path: EPath):
        self.final_name = tar_path.with_suffix(ITAR_SUFFIX)
        self.tmp_name = tar_path.with_suffix(ITAR_SUFFIX + ".tmp")
        self.itar = self.tmp_name.open("wb")

    def append(self, offset: int):
        self.itar.write(struct.pack("Q", offset))

    def close(self, finalize: bool = True):
        self.itar.close()
        if finalize:
            self.tmp_name.move(self.final_name)
        else:
            self.tmp_name.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(finalize=exc_val is None)


class SubFileReader(BinaryIO):
    """A file-like object that reads a subfile (i.e. offset, size defined portion) of a larger
    file."""

    def __init__(self, stream: BinaryIO, offset: int, size: int):
        self.offset = offset
        self._pos = 0
        self.size = size
        self.stream = stream
        self.stream.seek(self.offset)

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            n = self.size - self._pos
        else:
            n = min(n, self.size - self._pos)
        if n == 0:
            return b""
        read = self.stream.read(n)
        self._pos += len(read)
        return read

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self.size + offset
        else:
            raise ValueError("Invalid whence value")
        self._pos = max(0, min(self._pos, self.size))
        self.stream.seek(self.offset + self._pos)
        return self._pos

    def tell(self) -> int:
        return self._pos

    def __enter__(self) -> BinaryIO:
        return self

    def __exit__(
        self, exc_type: Type[BaseException], exc_val: BaseException, exc_tb: TracebackType
    ) -> None:
        self.close()

    def close(self) -> None:
        self.stream.close()

    def isatty(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False


def get_itar_byte_offset(
    path: Union[str, EPath],
    sample_offset: int = 0,
) -> int:
    """Gets the byte offset from sample offsets."""
    if sample_offset == 0:
        return 0
    with TarIndexReader(path) as itar:
        return itar[sample_offset]


@contextlib.contextmanager
def open_itar(path: Union[str, EPath], byte_offset: int = 0, byte_size: Optional[int] = None):
    """
    Open an indexed tarfile with offset and size.
    Args:
        path: Path to the tarfile to open
        byte_offset: Byte offset within the file
        byte_size: Size of the file to read

    Returns:
        The opened tarfile
    """
    path = EPath(path)

    # TODO: if tar file startswith(b"\x1f\x8b\x08") -> Seekable gzip file
    with path.open("rb") as f:
        if f.read(3) == b"\x1f\x8b\x08":
            # Open as seekable tgz
            raise ValueError("Seekable tgz not supported yet")

    if byte_offset != 0 or byte_size is not None:
        if byte_size is None:
            byte_size = path.size() - byte_offset
        with RetryReadStream(path) as stream:
            with SubFileReader(
                stream,
                offset=byte_offset,
                size=byte_size,
            ) as fileobj:
                with tarfile.open(fileobj=fileobj, mode="r|") as f:
                    yield f
    else:
        with RetryReadStream(path) as fileobj:
            with tarfile.open(fileobj=fileobj, mode="r|") as f:
                yield f
