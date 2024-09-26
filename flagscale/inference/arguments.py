import json
import argparse
import msgspec
import dataclasses

from enum import Enum
from inspect import isclass
from argparse import ArgumentTypeError
from typing import Any, Dict, List, NewType, Optional, Union, get_type_hints, get_args

from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

from flagscale.inference.core.sampling_params import SamplingParams


# This file is modified from
#  https://github.com/huggingface/transformers/blob/main/src/transformers/hf_argparser.py


ClassType = NewType("ClassType", Any)


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def ealy_stopping_str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == "never":
        return "Never"
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0, Never (case insensitive)."
        )


class FSArgumentParser(FlexibleArgumentParser):

    def __init__(self, dataclass_types: Union[ClassType, msgspec.Struct], **kwargs):
        super().__init__(**kwargs)

        self.parser_group_dict = {}
        for class_dtype in dataclass_types:
            if dataclasses.is_dataclass(class_dtype):
                self._add_dataclass_arguments(class_dtype)
            elif issubclass(class_dtype, msgspec.Struct):
                self._add_struct_arguments(class_dtype)
            else:
                raise TypeError(f"Unsupported type: {class_dtype}")

        self.add_flagscale_additional_args()

    def _add_dataclass_arguments(self, class_type: ClassType):
        assert class_type.__name__ == "EngineArgs"
        parser = self.add_argument_group(title=class_type.__name__)
        parser = class_type.add_cli_args(parser)
        self.parser_group_dict["vLLMEngine"] = parser

    @staticmethod
    def _parse_struct_field(parser: argparse.ArgumentParser, field: msgspec.structs.FieldInfo):
        SPICIAL_DTYPE_FIELDS = ["early_stopping", "logits_processors"]
        NOT_ARGUMENTS_FIELDS = ["output_text_buffer_length", "_all_stop_token_ids"]
        RENAME_FIELDS = {"seed": "sampling_seed"}

        if field.name in NOT_ARGUMENTS_FIELDS:
            return

        if field.name in RENAME_FIELDS:
            field.name = RENAME_FIELDS[field.name]

        field_name = f"--{field.name.replace('_', '-')}"
        origin_type = getattr(field.type, "__origin__", field.type)
        if origin_type is Union and field.name not in SPICIAL_DTYPE_FIELDS:
            if len(field.type.__args__) == 3:
                # filter [str,  List[str], NoneType] to [List[str], NoneType]
                assert type(List[field.type.__args__[0]]) == type(field.type.__args__[1])
                field.type.__args__ = field.type.__args__[1:]

            if len(field.type.__args__) != 2 or type(None) not in field.type.__args__:
                raise ValueError("Only `Union[X, NoneType]` (i.e., `Optional[X]`) is allowed for `Union`")
            if bool not in field.type.__args__:
                # filter `NoneType` in Union (except for `Union[bool, NoneType]`)
                field.type = (
                    field.type.__args__[0] if isinstance(None, field.type.__args__[1]) else field.type.__args__[1]
                )
                origin_type = getattr(field.type, "__origin__", field.type)

        kwargs = {}
        if isinstance(field.type, type) and issubclass(field.type, Enum):
            kwargs["choices"] = [x.value for x in field.type]
            kwargs["type"] = type(kwargs["choices"][0])
            if field.default is not msgspec.structs.NODEFAULT:
                kwargs["default"] = field.default
            else:
                kwargs["required"] = True
        elif field.type is bool or field.type == Optional[bool]:
            # Hack because type=bool in argparse does not behave as we want.
            kwargs["type"] = str2bool
            if field.type is bool or (field.default is not None and field.default is not msgspec.structs.NODEFAULT):
                default = False if field.default is msgspec.structs.NODEFAULT else field.default
                kwargs["default"] = default
                kwargs["nargs"] = "?"
                kwargs["const"] = True
        elif isclass(origin_type) and issubclass(origin_type, list):
            if hasattr(get_args(field.type)[0], "__args__"):
                kwargs["type"] = field.type.__args__[0].__args__[0]
                kwargs["action"] = "append"
            else:
                kwargs["type"] = field.type.__args__[0]

            kwargs["nargs"] = "+"
            if field.default_factory is not msgspec.structs.NODEFAULT:
                kwargs["default"] = field.default_factory()
            elif field.default is msgspec.structs.NODEFAULT:
                kwargs["required"] = True
        elif getattr(field.type, "__origin__", field.type) is Union:
            assert field.name in SPICIAL_DTYPE_FIELDS
            if field.name == "early_stopping":
                kwargs["type"] = ealy_stopping_str2bool
                if field.default is not None and field.default is not msgspec.structs.NODEFAULT:
                    default = "False" if field.default is msgspec.structs.NODEFAULT else field.default
                    kwargs["default"] = default
            elif field.name == "logits_processors":
                kwargs["type"] = str
                kwargs["nargs"] = "+"
                kwargs["default"] = field.default
        else:
            kwargs["type"] = json.loads if field.type is dict else field.type
            if field.default is not msgspec.structs.NODEFAULT:
                kwargs["default"] = field.default
            elif field.default_factory is not msgspec.structs.NODEFAULT:
                kwargs["default"] = field.default_factory()
            else:
                kwargs["required"] = True

        parser.add_argument(field_name, **kwargs)

    def _add_struct_arguments(self, class_type: msgspec.Struct):
        parser = self.add_argument_group(title=class_type.__name__)
        type_hints: Dict[str, type] = get_type_hints(class_type)
        for field in msgspec.structs.fields(class_type):
            field.type = type_hints[field.name]
            self._parse_struct_field(parser, field)

        self.parser_group_dict["SamplingParams"] = parser

    def add_flagscale_additional_args(self):

        parser = self.add_argument_group(title='flagscale')
        parser.add_argument("--prompts-path",
                            type=str,
                            default=None,
                            help="the text file contain the prompts")
        parser.add_argument("--prompts", 
                        nargs='*',
                        help="A list of prompts to generate completions for.")
        parser.add_argument("--negative-prompts",
                        nargs='*',
                        default=None,
                        help="A list of negative prompts")

        self.parser_group_dict["flagscale"] = parser

    def get_group(self, group_name):
        return self.parser_group_dict[group_name]


def parse_args(ignore_unknown_args=False):
    parser = FSArgumentParser((EngineArgs, SamplingParams), description='vLLM Inference')

    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    return args, parser
