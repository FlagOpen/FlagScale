# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import yaml

from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseCoreDataset, Sample
from megatron.energon.typed_converter import raw_to_instance
from megatron.energon.worker import WorkerConfig

T = TypeVar("T")

MAIN_FOLDER_NAME = ".nv-meta"


def load_config(
    path: Union[EPath, Dict[str, Any]],
    *,
    default_type: Optional[Type[T]] = None,
    strict: bool = True,
    default_kwargs: Optional[Dict[str, Any]] = None,
) -> T:
    """
    Loads a config from a file or directly from a dictionary.

    Args:
        path: Path to the config to load or a dictionary containing the config.
        default_type: If set, this is the type to use if no type is specified in the config.
        strict: If true, don't allow additional attributes in the config.
        default_kwargs: Default kwargs to use, will be overridden by the config.

    Returns:
        The instantiated type.
    """
    if isinstance(path, dict):
        data = path
    else:
        # Read the config from a file
        path = path.absolute()
        with path.open() as f:
            data: dict = yaml.safe_load(f)

    if default_kwargs is not None:
        new_data = default_kwargs.copy()
        new_data.update(data)
        data = new_data

    return raw_to_instance(data, default_type, strict=strict)


T_sample = TypeVar("T_sample", covariant=True)


def get_dataset_from_config(
    path: Union[EPath, Path, str],
    *,
    dataset_config: str = "dataset.yaml",
    split_config: str = "split.yaml",
    split_part: str = "train",
    training: bool = True,
    subflavor: Optional[str] = None,
    subflavors: Optional[Dict[str, Any]] = None,
    worker_config: Optional[WorkerConfig] = None,
    sample_type: Optional[Type[T_sample]] = None,
    **kwargs,
) -> BaseCoreDataset[T_sample]:
    """
    Gets a dataset from a config path.

    Args:
        path: Path to the folder where the `.nv-meta` folder is contained.
        dataset_config: Filename of the dataset config file (`path / '.nv-meta' / config`)
        split_config: Filename of the split config file (`path / '.nv-meta' / split_config`)
        split_part: Name of the split to load.
        training: If true, apply training randomization and loop the dataset.
        subflavor: Override the __subflavor__ property of each sample.
        subflavors: Merge-Override the __subflavors__ property of each sample.
        worker_config: If set, use this worker config instead of the default one.
        sample_type: Type of the samples to load, only used to ensure typing.
        **kwargs: Additional arguments to be passed to the dataset constructor.

    Returns:
        The instantiated dataset
    """
    path = EPath(path).absolute()
    if not (path / MAIN_FOLDER_NAME).is_dir():
        raise ValueError(
            f"Path {path} does not contain a {MAIN_FOLDER_NAME} folder. Did you forget to prepare "
            f"the dataset? Please check the documentation for an introduction to dataset "
            f"preparation."
        )
    dataset = load_config(
        path / MAIN_FOLDER_NAME / dataset_config,
        default_kwargs=dict(
            path=path,
            split_config=split_config,
            split_part=split_part,
            training=training,
            subflavor=subflavor,
            worker_config=worker_config,
            **kwargs,
        ),
    )
    if dataset.subflavors is None:
        dataset.subflavors = subflavors
    elif subflavors is not None:
        dataset.subflavors.update(subflavors)
    if sample_type is not None:
        assert issubclass(
            dataset.__sample_type__, sample_type
        ), f"Sample of type {dataset.__sample_type__} is not a subclass of {sample_type}."
    return dataset
