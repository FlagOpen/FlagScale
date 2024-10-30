# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Union

from megatron.energon.dataset_config import load_config
from megatron.energon.epathlib import EPath
from megatron.energon.metadataset.dataset_loader import DatasetLoader
from megatron.energon.metadataset.loader_interface import DatasetLoaderInterface
from megatron.energon.metadataset.metadataset import Metadataset


def load_dataset(
    path: Union[str, EPath, Path],
    **kwargs,
) -> DatasetLoaderInterface:
    """Loads a (meta)dataset."""

    if isinstance(path, dict):
        return load_config(
            path,
            default_type=Metadataset,
            strict=True,
            default_kwargs=dict(parent_path=EPath("/"), **kwargs),
        )
    path = EPath(path).absolute()
    if path.is_file():
        return load_config(
            path,
            default_type=Metadataset,
            strict=True,
            default_kwargs=dict(parent_path=path.parent, **kwargs),
        )
    else:
        return DatasetLoader(path=path, **kwargs)
