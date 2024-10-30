# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from typing import Any, Type, TypeVar, Union


def compact_str(
    value: Union[dict, list, str, int, bool, None],
    depth: int = 3,
    max_items: int = 10,
    max_str_len: int = 50,
) -> str:
    """
    Compact representation of a value as a string.

    Args:
        value: The value to compact
        depth: The maximum depth to compact
        max_items: The maximum number of items to show in a list or dict
        max_str_len: The maximum string length to show

    Returns: The printable string
    """
    if isinstance(value, dict):
        if depth <= 0:
            return "{...}"
        return (
            "{"
            + ", ".join(
                (
                    f"{k}: {v!r}"
                    if isinstance(k, str) and k.startswith("__")
                    else f"{k}: {compact_str(v, depth - 1, max_items, max_str_len)}"
                )
                for k, v in itertools.islice(value.items(), max_items)
            )
            + "}"
        )
    elif isinstance(value, list):
        if depth <= 0:
            return "[...]"
        return (
            "["
            + ", ".join(
                compact_str(v, depth - 1, max_items, max_str_len) for v in value[:max_items]
            )
            + "]"
        )
    elif isinstance(value, str):
        if len(value) > max_str_len:
            return repr(value[:max_str_len] + "...")
        return repr(value)
    else:
        return repr(value)


T = TypeVar("T")


class SampleException(ValueError):
    @classmethod
    def from_sample_key(cls: Type[T], sample_key: str) -> T:
        return cls(f"Sample {sample_key} failed")

    @classmethod
    def from_sample(cls: Type[T], sample: Any) -> T:
        return cls(f"Sample {compact_str(sample)} failed")


class FatalSampleError(SampleException):
    # This will not be handled by the error handler
    pass


SYSTEM_EXCEPTIONS = (
    SystemError,
    SyntaxError,
    ImportError,
    StopIteration,
    StopAsyncIteration,
    MemoryError,
    RecursionError,
    ReferenceError,
    NameError,
    UnboundLocalError,
    FatalSampleError,
)
