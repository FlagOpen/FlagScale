# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import re
from typing import Any, List, Union

from megatron.energon.errors import SampleException, compact_str

_field_split_re = re.compile(r"^(?P<field_name>[^[\]]+)(?P<access>(?:\[([^[\]]+)])*)$")


def split_field_access(field_desc: str) -> List[List[str]]:
    """
    Splits a field_map for access::

      'abcdef,ghi' -> [['abcdef'], ['ghi']]
      'abcdef[ghi]' -> [['abcdef', 'ghi']]
      'abcdef[ghi][jkl]' -> [['abcdef', 'ghi', 'jkl']]
    """
    options = field_desc.split(",")
    option_fields = []
    for option in options:
        match = _field_split_re.match(option)
        if match:
            option_fields.append(
                [match.group("field_name")]
                + [
                    access.lstrip("[").rstrip("]")
                    for access in match.group("access").split("][")
                    if access
                ]
            )
        else:
            option_fields.append([field_desc])
    return option_fields


class FieldAccessError(SampleException):
    pass


def _field_access(value: Union[dict, list, str, int, bool, None], field: List[str]) -> Any:
    """
    Accesses a (nested) field in the value.

    Args:
        value: The value to access
        field: The access instruction (e.g. `['field1', 'field2']` for
          `value['field1']['field2']`)

    Returns:
        The accessed value
    """
    try:
        if len(field) == 0:
            return value
        elif isinstance(value, dict):
            return _field_access(value[field[0]], field[1:])
        elif isinstance(value, list):
            return _field_access(value[int(field[0])], field[1:])
        else:
            raise FieldAccessError(
                f"Cannot access literal value {compact_str(value)} with {field!r}"
            )
    except FieldAccessError:
        raise
    except KeyError:
        raise FieldAccessError(f"Cannot access {'.'.join(field)!r} in {compact_str(value)}")


def field_access(value: Union[dict, list, str, int, bool, None], field: List[List[str]]) -> Any:
    """
    Accesses a (nested) field in the value.

    Args:
        value: The value to access
        field: The access instruction (e.g. `[['field1', 'field2']]` for
          `value['field1']['field2']`, or `[['field1'], ['field2']]` for value.get('field1', value['field2'])`)

    Returns:
        The accessed value
    """
    for f in field[:-1]:
        try:
            return _field_access(value, f)
        except (KeyError, ValueError, IndexError):
            pass
    return _field_access(value, field[-1])
