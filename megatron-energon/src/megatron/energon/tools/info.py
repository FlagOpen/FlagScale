# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from io import StringIO
from typing import Type
import click
from importlib import import_module

from megatron.energon.dataset_config import MAIN_FOLDER_NAME
from megatron.energon.epathlib import EPath



import click
import yaml

from megatron.energon.flavors.base_dataset import BaseCoreDataset, Sample

fmt = """
* <dataset-name>: `{path}` ({samples_count:,} samples, {samples_size} in {shards_count} shards)
  * Type: {{py:class}}`{class_name} <{class_name}>`
  * Sample Type: {{py:class}}`{sample_name} <{sample_name}>`
  * Default Splits:
{splits_str}
"""

split_fmt = """    * `{split_name}`: {split_ratio:.0f}%, {split_samples_count:,} samples in {split_shards_count} shards
"""


def fmt_size(size: int) -> str:
    keys = ["B", "KiB", "MiB", "GiB", "TiB"]
    for key in keys:
        if size < 1024:
            return f"{size:.2f} {key}"
        size /= 1024
    return f"{size:.2f} TiB"


@click.command(name="info")
@click.argument(
    "path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=EPath),
)
@click.option(
    "--split-config", default="split.yaml", help="Split config file name", show_default=True
)
@click.option(
    "--dataset-config", default="dataset.yaml", help="Dataset config file name", show_default=True
)
def command(
    path: EPath,
    split_config: str,
    dataset_config: str,
):
    """
    Print information about a dataset, formatted to be pasted in `existing_datasets.md`.
    """
    ds_config = yaml.safe_load(StringIO((path / MAIN_FOLDER_NAME / dataset_config).read_text()))
    info_config = yaml.safe_load(StringIO((path / MAIN_FOLDER_NAME / ".info.yaml").read_text()))
    split_config = yaml.safe_load(StringIO((path / MAIN_FOLDER_NAME / split_config).read_text()))
    samples_count = sum(info_config["shard_counts"].values())
    strmodule = ds_config["__module__"].replace("nvgpt4.data", "megatron.energon")
    module = import_module(strmodule)
    cls: Type[BaseCoreDataset] = getattr(module, ds_config["__class__"])
    class_name = cls.__name__
    class_fullname = cls.__qualname__
    sample_type: Type[Sample] = cls.__sample_type__
    sample_name = sample_type.__name__
    sample_fullname = sample_type.__qualname__

    def srt_key(pair):
        try:
            return ("train", "val", "test").index(pair[0])
        except ValueError:
            return 3

    splits_str = "".join(
        split_fmt.format(
            split_name=split_name,
            split_ratio=round(
                100
                * sum(info_config["shard_counts"][shard] for shard in split_parts)
                / samples_count,
                2,
            ),
            split_samples_count=sum(info_config["shard_counts"][shard] for shard in split_parts),
            split_shards_count=len(split_parts),
        )
        for split_name, split_parts in sorted(split_config["split_parts"].items(), key=srt_key)
    )
    print(
        fmt.format(
            path=str(path),
            samples_count=samples_count,
            samples_size=fmt_size(
                sum(
                    (path / split_name).size()
                    for split_name in info_config["shard_counts"].keys()
                )
            ),
            shards_count=len(info_config["shard_counts"]),
            class_name=class_name,
            class_fullname=class_fullname,
            sample_name=sample_name,
            sample_fullname=sample_fullname,
            splits_str=splits_str,
        )
    )
