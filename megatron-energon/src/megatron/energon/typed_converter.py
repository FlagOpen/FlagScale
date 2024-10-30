# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
"""Provides methods for converting typed objects to json objects and vice versa."""

import dataclasses
import importlib
import inspect
import json
import re
import typing
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type, TypeVar, Union

from megatron.energon.epathlib import EPath


class JsonValueError(ValueError):
    """Internal error class, for better error messages."""

    def __init__(
        self, msg: str, expected_type: Type, value: Any, path: str, stage: Tuple[int, ...]
    ) -> None:
        """New error"""
        super().__init__(msg)
        self.expected_type = expected_type
        self.value = value
        self.path = path
        self.stage = stage


class _MissingValue:
    """Internal class representing a missing value (i.e. a field which has not been specified,
    opposed to explicit `None`) for better error messages."""

    def __repr__(self) -> str:
        return "missing value"


_missing_value = _MissingValue()

TType = TypeVar("TType")


def _check_instance_type(cls, inst_type: Type) -> bool:
    """Check if a class is an instance of a type."""
    if inst_type is None:
        return True
    elif typing.get_origin(inst_type) is not None:
        org = typing.get_origin(inst_type)
        if org is Union:
            for check_type in typing.get_args(inst_type):
                if _check_instance_type(cls, check_type):
                    return True
        elif isinstance(org, type) and issubclass(cls, org):
            return True
    elif inst_type is Any:
        return True
    else:
        return not issubclass(inst_type, type) or issubclass(cls, inst_type)


def raw_to_instance(kwargs: dict, inst_type: Type[TType], strict: bool = True) -> TType:
    """
    Try to import and instantiate a class from a dict with "__module__" and "__class__" keys.

    Args:
        kwargs: The dict to parse
        inst_type: Expected return type, used if type is not specified in the kwargs
        strict: If true, don't allow additional attributes

    Returns:
        Instantiated class
    """
    kwargs = kwargs.copy()
    module_name = kwargs.pop("__module__", None)
    class_name = kwargs.pop("__class__", None)
    if module_name is None or class_name is None:
        cls = inst_type
    else:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            try_energon_module = importlib.import_module("megatron.energon", package=None)
            if hasattr(try_energon_module, class_name):
                module = try_energon_module
            else:
                raise

        cls = typing.cast(Type[TType], getattr(module, class_name))
        if isinstance(cls, type):
            assert _check_instance_type(cls, inst_type), f"Expected {inst_type}, got {cls}"
        elif not callable(cls):
            raise ValueError(f"Expected a class or a callable, got {cls}")
    inst = safe_call_function(kwargs, cls, strict=strict, allow_imports=True)
    if not isinstance(cls, type):
        assert _check_instance_type(
            type(inst), inst_type
        ), f"Expected {inst_type}, got {type(inst)}"
    return inst


def raw_to_typed(  # noqa: C901
    raw_data: Union[dict, list, str, int, bool, float, None],
    inst_type: Type[TType],
    strict: bool = False,
    allow_imports: bool = False,
    _path: str = "root",
    _stage: Tuple[int, ...] = (),
) -> TType:
    """
    Converts raw data (i.e. dicts, lists and primitives) to typed objects (like
    `NamedTuple` or `dataclasses.dataclass`). Validates that python typing matches.

    Usage::

        class MyNamedTuple(NamedTuple):
            x: int
            y: str

        assert raw_to_typed({'x': int, 'y': "foo"}, MyNamedTuple) == MyNamedTuple(x=5, y="foo")

    Args:
        raw_data: The raw (e.g. json) data to be made as `inst_type`
        inst_type: The type to return
        strict: If true, don't allow additional attributes
        allow_imports: If true, parse '__module__' and '__class__' attributes to allow explicit
            instantiation of types
        _path: (internal for recursive call) The path to the object being converted from the root
        _stage: (internal for recursive call) Numbers representing the position of the current
            object being converted from the root

    Returns:
        The input data as `inst_type`.
    """
    type_name = getattr(inst_type, "__name__", repr(inst_type))
    if raw_data is _missing_value:
        raise JsonValueError(
            f"Missing value at {_path}",
            inst_type,
            raw_data,
            _path,
            _stage,
        )
    elif inst_type in (str, int, float, bool, None, type(None)):
        # Literal types or missing data
        if not isinstance(raw_data, inst_type) and not (
            isinstance(raw_data, int) and inst_type == float
        ):
            raise JsonValueError(
                f"Type does not match, expected {type_name} at {_path}, got {raw_data!r}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
        return raw_data
    elif inst_type is Any:
        if (
            allow_imports
            and isinstance(raw_data, dict)
            and "__module__" in raw_data
            and "__class__" in raw_data
        ):
            return raw_to_instance(raw_data, inst_type, strict=strict)
        # Any
        return raw_data
    elif typing.get_origin(inst_type) is Literal:
        # Literal[value[, ...]]
        values = typing.get_args(inst_type)
        if raw_data not in values:
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {raw_data!r}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
        return raw_data
    elif typing.get_origin(inst_type) is Union:
        # Union[union_types[0], union_types[1], ...]
        union_types = typing.get_args(inst_type)
        if None in union_types:
            # Fast Optional path
            if raw_data is None:
                return None
        best_inner_error: Optional[JsonValueError] = None
        multiple_matching = False
        for subtype in union_types:
            try:
                return raw_to_typed(
                    raw_data,
                    subtype,
                    strict,
                    allow_imports,
                    f"{_path} -> {getattr(subtype, '__name__', repr(subtype))}",
                    _stage + (1,),
                )
            except JsonValueError as err:
                if best_inner_error is None or err.stage > best_inner_error.stage:
                    best_inner_error = err
                    multiple_matching = False
                elif err.stage == best_inner_error.stage:
                    multiple_matching = True
                continue
        if best_inner_error is None or multiple_matching:
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {raw_data!r}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
        else:
            raise best_inner_error
    elif (
        isinstance(inst_type, type)
        and issubclass(inst_type, tuple)
        and hasattr(inst_type, "__annotations__")
    ):
        # class MyClass(NamedTuple): ...
        if not isinstance(raw_data, dict):
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {raw_data!r}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
        if getattr(inst_type, "__dash_keys__", "False"):
            raw_data = {key.replace("-", "_"): val for key, val in raw_data.items()}
        defaults = getattr(inst_type, "_field_defaults", {})
        kwargs = {
            field_name: raw_to_typed(
                raw_data.get(field_name, defaults.get(field_name, _missing_value)),
                field_type,
                strict,
                allow_imports,
                f"{_path} -> {type_name}:{field_name}",
                _stage + (idx,),
            )
            for idx, (field_name, field_type) in enumerate(inst_type.__annotations__.items())
        }
        if strict and not set(raw_data).issubset(inst_type.__annotations__):
            raise JsonValueError(
                f"Additional attributes for {type_name} at {_path}, got {raw_data!r}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
        try:
            return inst_type(**kwargs)
        except BaseException:
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {raw_data!r}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
    elif dataclasses.is_dataclass(inst_type):
        # dataclass
        if not isinstance(raw_data, dict):
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {raw_data!r}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
        kwargs = {
            field.name: raw_to_typed(
                raw_data.get(
                    field.name,
                    (
                        (
                            _missing_value
                            if field.default_factory is dataclasses.MISSING
                            else field.default_factory()
                        )
                        if field.default is dataclasses.MISSING
                        else field.default
                    ),
                ),
                field.type,
                strict,
                allow_imports,
                f"{_path} -> {type_name}:{field.name}",
                _stage + (idx,),
            )
            for idx, field in enumerate(dataclasses.fields(inst_type))
            if field.init
        }
        if strict and not set(raw_data).issubset(
            field.name for field in dataclasses.fields(inst_type) if field.init
        ):
            raise JsonValueError(
                f"Additional attributes for {type_name} at {_path}, got {raw_data!r}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
        try:
            return inst_type(**kwargs)
        except BaseException:
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {raw_data!r}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
    elif typing.get_origin(inst_type) is list:
        # List[inner_type]
        (inner_type,) = typing.get_args(inst_type)
        if not isinstance(raw_data, list):
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {raw_data!r}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
        return [
            raw_to_typed(
                val, inner_type, strict, allow_imports, f"{_path} -> {idx}", _stage + (idx,)
            )
            for idx, val in enumerate(raw_data)
        ]
    elif typing.get_origin(inst_type) is set:
        # Set[inner_type]
        (inner_type,) = typing.get_args(inst_type)
        if not isinstance(raw_data, list):
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {raw_data!r}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
        res = set(
            raw_to_typed(
                val, inner_type, strict, allow_imports, f"{_path} -> {idx}", _stage + (idx,)
            )
            for idx, val in enumerate(raw_data)
        )
        if len(res) != len(raw_data):
            raise JsonValueError(
                f"Duplicate element at {_path}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
        return res
    elif typing.get_origin(inst_type) is tuple:
        # Tuple[inner_types[0], inner_types[1], ...] or Tuple[inner_types[0], Ellipsis/...]
        inner_types = typing.get_args(inst_type)
        if not isinstance(raw_data, list):
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {raw_data!r}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
        if len(inner_types) == 2 and inner_types[1] is Ellipsis:
            # Tuple of arbitrary length, all elements same type
            # Tuple[inner_types[0], Ellipsis/...]
            return tuple(
                raw_to_typed(
                    val, inner_types[0], strict, allow_imports, f"{_path} -> {idx}", _stage + (idx,)
                )
                for idx, val in enumerate(raw_data)
            )
        else:
            # Fixed size/typed tuple
            # Tuple[inner_types[0], inner_types[1], ...]
            if len(raw_data) != len(inner_types):
                raise JsonValueError(
                    f"Expected {type_name} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
            return [
                raw_to_typed(
                    val, inner_type, strict, allow_imports, f"{_path} -> {idx}", _stage + (idx,)
                )
                for idx, (val, inner_type) in enumerate(zip(raw_data, inner_types))
            ]
    elif typing.get_origin(inst_type) is dict:
        # Dict[str, value_type]
        key_type, value_type = typing.get_args(inst_type)
        assert key_type is str
        if not isinstance(raw_data, dict):
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {raw_data!r}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
        return {
            key: raw_to_typed(
                val, value_type, strict, allow_imports, f"{_path} -> {key!r}", _stage + (idx,)
            )
            for idx, (key, val) in enumerate(raw_data.items())
        }
    elif inst_type in (dict, list):
        # dict, list (no subtyping)
        if not isinstance(raw_data, inst_type):
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {raw_data!r}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
        return raw_data
    elif inst_type == EPath:
        if isinstance(raw_data, str):
            return EPath(raw_data)
        elif not isinstance(raw_data, EPath):
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {raw_data!r}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
        return raw_data
    elif (
        allow_imports
        and isinstance(raw_data, dict)
        and "__module__" in raw_data
        and "__class__" in raw_data
    ):
        return raw_to_instance(raw_data, inst_type, strict=strict)
    else:
        return raw_data


def safe_call_function(
    raw_data: Union[dict, list, str, int, bool, float, None],
    fn: Callable[..., TType],
    strict: bool = False,
    allow_imports: bool = False,
) -> TType:
    """
    Converts raw data (i.e. dicts, lists and primitives) to typed call arguments.
    Validates that python typing matches.

    Usage::

        def fn(arg1: float, arg2: MyType, arg3) -> Any:
            assert isinstance(arg1, float)
            assert isinstance(arg2, MyType)

        fn(3.141, MyType(), None)

    Args:
        raw_data: The raw (e.g. json) data to be made as `inst_type`
        fn: The function to call with the converted data
        strict: If true, don't allow additional attributes
        allow_imports: If true, allow instantiating objects by specifying __module__ and __class__.

    Returns:
        The return value of `fn`
    """
    parameters = list(inspect.signature(fn).parameters.items())
    if inspect.isclass(fn):
        init_sig = getattr(fn, "__init__", None)
        if init_sig is not None:
            parameters = list(inspect.signature(init_sig).parameters.items())[1:]
    args = []
    kwargs = {}
    if isinstance(raw_data, dict):
        unused_args = raw_data.copy()
        for idx, (key, param) in enumerate(parameters):
            t = Any if param.annotation is inspect.Parameter.empty else param.annotation
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                if param.default is inspect.Parameter.empty and key not in unused_args:
                    raise ValueError(f"Missing required argument {key!r} for {fn}")
                kwargs[key] = raw_to_typed(
                    unused_args.pop(key, param.default),
                    t,
                    strict,
                    allow_imports,
                    _path=key,
                    _stage=(idx,),
                )
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                for arg_key, arg_val in unused_args.items():
                    kwargs[arg_key] = raw_to_typed(
                        arg_val, t, strict, allow_imports, _path=key, _stage=(idx,)
                    )
                unused_args.clear()
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                # No way to pass positional arguments
                pass
            elif param.kind == inspect.Parameter.POSITIONAL_ONLY:
                # No way to pass positional arguments
                raise RuntimeError(f"Unsupported positional only argument {key!r}")
            else:
                raise RuntimeError(f"Unknown parameter kind {param.kind!r}")
        if strict and len(unused_args) > 0:
            raise ValueError(f"Unexpected arguments: {unused_args!r}")
    elif isinstance(raw_data, list):
        unused_args = raw_data.copy()
        for idx, (key, param) in enumerate(parameters):
            t = Any if param.annotation is inspect.Parameter.empty else param.annotation
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                if param.default is inspect.Parameter.empty and len(unused_args) == 0:
                    raise ValueError(
                        f"Missing required positional-only argument {key!r} at index {idx}"
                    )
                args.append(
                    raw_to_typed(
                        unused_args.pop(), t, strict, allow_imports, _path=key, _stage=(idx,)
                    )
                )
            elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                if param.default is inspect.Parameter.empty and len(unused_args) == 0:
                    raise ValueError(f"Missing required positional argument {key!r} at index {idx}")
                if len(unused_args) == 0:
                    arg_val = param.default
                else:
                    arg_val = unused_args.pop()
                args.append(
                    raw_to_typed(arg_val, t, strict, allow_imports, _path=key, _stage=(idx,))
                )
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                for arg_val in unused_args:
                    args.append(
                        raw_to_typed(arg_val, t, strict, allow_imports, _path=key, _stage=(idx,))
                    )
                unused_args.clear()
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                # No way to pass keyword arguments
                pass
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise RuntimeError(f"Unsupported keyword-only argument {key!r}")
            else:
                raise RuntimeError(f"Unknown parameter kind {param.kind!r}")
        if strict and len(unused_args) > 0:
            raise ValueError(f"Unexpected arguments: {unused_args!r}")
    else:
        raise ValueError(
            f"Cannot call function with raw data of type {type(raw_data)!r}, require list or dict"
        )
    return fn(*args, **kwargs)


float_pattern = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")


def _split_dict_keys(dct: Dict[str, Any]) -> Dict[str, Any]:
    """Splits the given dict keys by first '.' to subdicts."""
    res = {}
    for key, value in dct.items():
        if "." in key:
            outer_key, _, inner_key = key.partition(".")
            if outer_key in res:
                if not isinstance(res[outer_key], dict):
                    raise ValueError(f"Cannot combine {outer_key!r} with {res!r}")
                res[outer_key][inner_key] = value
            else:
                res[outer_key] = {inner_key: value}
        else:
            if key in res:
                raise ValueError(f"Cannot combine {key!r} with {res!r}")
            res[key] = value

    return res


def _isinstance_deep(val: Any, tp_chk: Type) -> bool:
    """Verifies if the given value is an instance of the tp_chk, allowing for typing extensions."""
    if tp_chk is Any:
        return True
    elif typing.get_origin(tp_chk) is Literal:
        (value,) = typing.get_args(val)
        return val == value
    elif typing.get_origin(tp_chk) is list:
        (inner_type,) = typing.get_args(val)
        return isinstance(val, list) and all(_isinstance_deep(v, inner_type) for v in val)
    elif typing.get_origin(tp_chk) is tuple:
        inner_types = typing.get_args(val)
        if len(inner_types) == 2 and inner_types[1] == Ellipsis:
            return isinstance(val, tuple) and all(_isinstance_deep(v, inner_types[0]) for v in val)
        else:
            return (
                isinstance(val, tuple)
                and len(val) == len(inner_types)
                and all(_isinstance_deep(v, inner_type) for v, inner_type in zip(val, inner_types))
            )
    elif typing.get_origin(tp_chk) is dict:
        key_type, value_type = typing.get_args(val)
        return isinstance(val, dict) and all(
            _isinstance_deep(k, key_type) and _isinstance_deep(v, value_type)
            for k, v in val.items()
        )
    else:
        return isinstance(val, tp_chk)


def override(  # noqa: C901
    value: TType,
    overrides: Any,
    strict: bool = False,
    inst_type: Optional[Type[TType]] = None,
    allow_imports: bool = False,
    _path: str = "root",
    _stage: Tuple[int, ...] = (),
) -> TType:
    """
    Allows overriding values of a typed object using environment config.
    Allows overriding single config variables, or whole objects.

    Examples::

        class MyNamedTuple(NamedTuple):
            x: int
            y: str

        class MyNested(NamedTuple):
            nested: MyNamedTuple

        assert override(
            MyNested(nested=MyNamedTuple(x=42, y="foo")),
            {'nested.x': 5},
        ) == MyNested(nested=MyNamedTuple(x=5, y="foo"))
        assert override(
            MyNested(nested=MyNamedTuple(x=42, y="foo")),
            {'nested': '{"x": 5, "y": "bar"}'},
        ) == MyNested(nested=MyNamedTuple(x=5, y="bar"))

    Args:
        value: The base value to override.
        overrides: The overrides to apply
        strict: If true, no additional keys are allowed
        inst_type: If given, validate against this base type instead of the type of `value`.
        allow_imports: If true, allow instantiating types with dicts of __module__ and __class__.
        _path: Internal: The path to the current value.
        _stage: Internal: The current stage of the override.

    Returns:
        Same type as the input object (or `inst_type` if set), copied and updated from the
        overrides.
    """
    if inst_type is None:
        inst_type = type(value)
    type_name = getattr(inst_type, "__name__", repr(inst_type))
    if inst_type in (str, int, float, bool, None, type(None)):
        # Literal types
        if inst_type in (None, type(None)) and overrides == "None":
            overrides = None
        elif inst_type == bool and overrides in ("True", "true", "1", "False", "false", "0"):
            overrides = overrides in ("True", "true", "1")
        elif inst_type in (int, float) and isinstance(overrides, str):
            overrides = inst_type(overrides)
        if not isinstance(overrides, inst_type) and not (
            isinstance(overrides, int) and inst_type == float
        ):
            raise JsonValueError(
                f"Type does not match, expected {type_name} at {_path}, got {overrides!r}",
                inst_type,
                overrides,
                _path,
                _stage,
            )
        return overrides
    elif inst_type is Any:
        # Any
        if isinstance(overrides, str):
            if overrides.isnumeric():
                return int(overrides)
            elif overrides == "True":
                return True
            elif overrides == "False":
                return True
            return overrides
        if isinstance(value, (dict, list, tuple)):
            # Merge with dict, list, str
            return override(value, overrides, strict, type(value), allow_imports, _path, _stage)
        raise JsonValueError(
            f"Expected {type_name} at {_path}, got {overrides!r}",
            inst_type,
            overrides,
            _path,
            _stage,
        )
    elif typing.get_origin(inst_type) is Literal:
        # Literal[value]
        (value,) = typing.get_args(inst_type)
        if value != overrides:
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {overrides!r}",
                inst_type,
                overrides,
                _path,
                _stage,
            )
        return value
    elif typing.get_origin(inst_type) is Union:
        # Union[union_types[0], union_types[1], ...]
        union_types = typing.get_args(inst_type)
        if isinstance(overrides, str):
            for subtype in union_types:
                if subtype is None and overrides == "None":
                    return None
                elif subtype is bool:
                    if overrides == "True":
                        return True
                    elif overrides == "False":
                        return False
                elif subtype is int and overrides.strip().isnumeric():
                    return int(overrides)
                elif subtype is str:
                    return overrides
                elif subtype is float and float_pattern.fullmatch(overrides):
                    return float(overrides)
            if overrides.lstrip().startswith("{") or overrides.lstrip().startswith("["):
                overrides = json.loads(overrides)
            return raw_to_typed(
                overrides,
                inst_type,
                strict,
                allow_imports,
                _path,
                _stage,
            )
        for subtype in union_types:
            if _isinstance_deep(value, subtype):
                return override(
                    value,
                    overrides,
                    strict,
                    subtype,
                    allow_imports,
                    f"{_path} -> {getattr(subtype, '__name__', repr(subtype))}",
                    _stage + (1,),
                )
        raise JsonValueError(
            f"Expected {type_name} at {_path}, existing is {value!r} which is invalid",
            inst_type,
            value,
            _path,
            _stage,
        )
    elif (
        isinstance(inst_type, type)
        and issubclass(inst_type, tuple)
        and hasattr(inst_type, "__annotations__")
    ):
        # class MyClass(NamedTuple): ...
        if not isinstance(overrides, (dict, str)):
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {overrides!r}",
                inst_type,
                overrides,
                _path,
                _stage,
            )
        if isinstance(overrides, str):
            return raw_to_typed(
                json.loads(overrides),
                inst_type,
                strict,
                allow_imports,
                _path,
                _stage,
            )
        local_overrides = _split_dict_keys(overrides)
        if getattr(inst_type, "__dash_keys__", "False"):
            local_overrides = {key.replace("-", "_"): val for key, val in local_overrides.items()}
        kwargs = {
            field_name: (
                override(
                    getattr(value, field_name),
                    local_overrides.pop(field_name),
                    strict,
                    field_type,
                    allow_imports,
                    f"{_path} -> {type_name}:{field_name}",
                    _stage + (idx,),
                )
                if field_name in local_overrides
                else getattr(value, field_name)
            )
            for idx, (field_name, field_type) in enumerate(inst_type.__annotations__.items())
        }
        if strict and len(local_overrides) != 0:
            raise JsonValueError(
                f"Invalid config keys {', '.join(local_overrides.keys())} for {type_name} at "
                f"{_path}",
                inst_type,
                overrides,
                _path,
                _stage,
            )
        try:
            return inst_type(**kwargs)
        except BaseException:
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {overrides!r}",
                inst_type,
                overrides,
                _path,
                _stage,
            )
    elif dataclasses.is_dataclass(inst_type):
        # dataclass
        if not isinstance(overrides, (dict, str)):
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {overrides!r}",
                inst_type,
                overrides,
                _path,
                _stage,
            )
        if isinstance(overrides, str):
            return raw_to_typed(
                json.loads(overrides),
                inst_type,
                strict,
                allow_imports,
                _path,
                _stage,
            )
        local_overrides = _split_dict_keys(overrides)
        if getattr(inst_type, "__dash_keys__", "False"):
            local_overrides = {key.replace("-", "_"): val for key, val in local_overrides.items()}
        kwargs = {
            field.name: (
                override(
                    getattr(value, field.name),
                    local_overrides.pop(field.name),
                    strict,
                    field.type,
                    allow_imports,
                    f"{_path} -> {type_name}:{field.name}",
                    _stage + (idx,),
                )
                if field.name in local_overrides
                else getattr(value, field.name)
            )
            for idx, field in enumerate(dataclasses.fields(inst_type))
            if field.init
        }
        if strict and len(local_overrides) != 0:
            raise JsonValueError(
                f"Invalid config keys {', '.join(local_overrides.keys())} for {type_name} at "
                f"{_path}",
                inst_type,
                overrides,
                _path,
                _stage,
            )
        try:
            return inst_type(**kwargs)
        except BaseException:
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {overrides!r}",
                inst_type,
                overrides,
                _path,
                _stage,
            )
    elif (
        typing.get_origin(inst_type) is list
        or typing.get_origin(inst_type) is tuple
        or inst_type in (list, tuple)
    ):
        # List[inner_type] or Tuple[inner_type, Ellipsis] or
        # Tuple[inner_type[0], inner_type[1], ...]
        if inst_type == list:
            inner_type = Any
            inner_types = []
            cls = list
        elif inst_type == tuple:
            inner_type = Any
            inner_types = []
            cls = tuple
        elif typing.get_origin(inst_type) is list:
            (inner_type,) = typing.get_args(inst_type)
            inner_types = []
            cls = list
        else:
            inner_types = typing.get_args(inst_type)
            if len(inner_types) == 2 and inner_types[1] is Ellipsis:
                inner_type = inner_types[0]
            else:
                inner_type = None
            cls = tuple
        if not isinstance(overrides, (dict, str)):
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {overrides!r}",
                inst_type,
                overrides,
                _path,
                _stage,
            )
        if isinstance(overrides, str):
            return raw_to_typed(
                json.loads(overrides),
                inst_type,
                strict,
                allow_imports,
                _path,
                _stage,
            )
        local_overrides = _split_dict_keys(overrides)
        if not all(key.isnumeric() for key in local_overrides.keys()):
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {overrides!r}, expected integer keys",
                inst_type,
                overrides,
                _path,
                _stage,
            )
        local_overrides_int = {int(key): value for key, value in local_overrides.items()}
        new_max_idx = max(local_overrides_int.keys())
        original_max_idx = len(value)
        if inner_type is None and new_max_idx >= len(inner_types):
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {overrides!r}, index {new_max_idx} out of "
                f"bounds",
                inst_type,
                overrides,
                _path,
                _stage,
            )
        for i in range(original_max_idx, new_max_idx):
            if i not in local_overrides_int:
                raise JsonValueError(
                    f"Expected {type_name} at {_path}, got {overrides!r}, missing value for index "
                    f"{i}",
                    inst_type,
                    overrides,
                    _path,
                    _stage,
                )
        return cls(
            (
                override(
                    value[idx],
                    local_overrides_int[idx],
                    strict,
                    inner_type,
                    allow_imports,
                    f"{_path} -> {idx}",
                    _stage + (idx,),
                )
                if idx in local_overrides_int
                else value[idx]
            )
            for idx in range(max(new_max_idx + 1, original_max_idx))
        )
    elif typing.get_origin(inst_type) is dict or inst_type == dict:
        # Dict[str, value_type]
        if inst_type == dict:
            value_type = Any
        else:
            key_type, value_type = typing.get_args(inst_type)
            assert key_type is str
        if not isinstance(overrides, (dict, str)):
            raise JsonValueError(
                f"Expected {type_name} at {_path}, got {overrides!r}",
                inst_type,
                overrides,
                _path,
                _stage,
            )
        if isinstance(overrides, str):
            return raw_to_typed(
                json.loads(overrides),
                inst_type,
                strict,
                allow_imports,
                _path,
                _stage,
            )
        local_overrides = _split_dict_keys(overrides)
        if getattr(inst_type, "__dash_keys__", "False"):
            local_overrides = {key.replace("-", "_"): val for key, val in local_overrides.items()}
        res = {
            key: (
                override(
                    subvalue,
                    local_overrides.pop(key),
                    strict,
                    value_type,
                    allow_imports,
                    f"{_path} -> {type_name}:{key!r}",
                    _stage + (idx,),
                )
                if key in local_overrides
                else subvalue
            )
            for idx, (key, subvalue) in value.items()
        }
        for key, val in local_overrides.items():
            if not isinstance(val, str):
                raise JsonValueError(
                    f"Expected new {type_name} at {_path} -> {type_name}:{key!r}, got {val!r}",
                    inst_type,
                    overrides,
                    _path,
                    _stage,
                )
            res[key] = raw_to_typed(
                json.loads(val),
                value_type,
                strict,
                allow_imports,
                f"{_path} -> {type_name}:{key!r}",
                _stage + (len(res),),
            )
        return res
    else:
        raise RuntimeError(f"Unknown type {inst_type}")


def to_json_object(obj: Any) -> Any:
    """
    Converts the given object to a json object.

    Args:
        obj: The object to convert

    Returns:
        The json-like object.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        # Literal types
        return obj
    elif isinstance(obj, tuple) and hasattr(obj, "__annotations__"):
        # class MyClass(NamedTuple): ...
        return {
            field_name: to_json_object(getattr(obj, field_name))
            for field_name in obj.__annotations__.keys()
        }
    elif dataclasses.is_dataclass(obj):
        # dataclass
        return {
            field.name: to_json_object(getattr(obj, field.name))
            for field in dataclasses.fields(obj)
            if field.init
        }
    elif isinstance(obj, (list, tuple)):
        return [to_json_object(val) for val in obj]
    elif isinstance(obj, dict):
        return {key: to_json_object(val) for key, val in obj.items()}
    else:
        raise RuntimeError(f"Unknown type {type(obj)}")
