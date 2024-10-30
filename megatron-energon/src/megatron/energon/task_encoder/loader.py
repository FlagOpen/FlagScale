# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, TypeVar, Union

from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseCoreDataset, SavableDataset
from megatron.energon.metadataset import load_dataset
from megatron.energon.task_encoder.base import DefaultTaskEncoder, TaskEncoder, WorkerConfig

T = TypeVar("T", covariant=True)


def _split_kwargs(kwargs: dict) -> dict:
    loader_kwargs = {}
    try:
        loader_kwargs["split_part"] = kwargs.pop("split_part")
    except KeyError:
        pass
    try:
        loader_kwargs["subflavor"] = kwargs.pop("subflavor")
    except KeyError:
        pass
    try:
        loader_kwargs["dataset_config"] = kwargs.pop("dataset_config")
    except KeyError:
        pass
    try:
        loader_kwargs["split_config"] = kwargs.pop("split_config")
    except KeyError:
        pass
    return loader_kwargs


def get_train_dataset(
    path: Union[str, EPath, Path],
    *,
    split_part: Union[Literal["train"], str] = "train",
    worker_config: WorkerConfig,
    batch_size: int,
    batch_drop_last: bool = False,
    packing_buffer_size: Optional[int] = None,
    shuffle_buffer_size: Optional[int],
    max_samples_per_sequence: Optional[int],
    virtual_epoch_length: int = 0,
    shuffle_over_epochs_multiplier: int = 1,
    task_encoder: TaskEncoder[Any, Any, Any, T] = DefaultTaskEncoder(),
    **kwargs,
) -> SavableDataset[T]:
    """
    Get training data loader with sensible defaults. See `get_dataset` for more details.

    The following recipe will be used:
      - :func:`megatron.energon.dataset_config.get_dataset_from_config` (loads the raw dataset)
      - `task_encoder.encode_sample`
      - (:class:`megatron.energon.MixDataset` if mixing)
      - :class:`megatron.energon.BatchDataset` with `task_encoder.batch` for collation
      - `task_encoder.encode_batch`
      - :class:`megatron.energon.EpochizeDataset` (if `virtual_epoch_length` is set)

    Args:
        path: Path to the dataset.
        split_part: Default split part to use.
        worker_config: Worker configuration to use.
        batch_size: Size of a batch
        batch_drop_last: If true, drop the last batch if it is smaller than `batch_size`.
        shuffle_buffer_size: Size of the sample shuffle buffer (before task encoding).
        max_samples_per_sequence: If set, limit the number of samples per sample-sequence to this.
        virtual_epoch_length: If set, the dataset will be epochized to this length (=iterating
            will be suspended and the for-loop returns, next for-loop continues iterating).
            Otherwise, the dataset will loop infinitely.
        shuffle_over_epochs_multiplier: Shuffle the shards over this many epochs.
        task_encoder: Task encoder to use.
        **kwargs: Additional arguments to the dataset constructor.

    Returns:
        The dataloader.
    """

    loader = load_dataset(path, **_split_kwargs(kwargs))
    datasets = loader.get_datasets(
        training=True,
        split_part=split_part,
        worker_config=worker_config,
        max_samples_per_sequence=max_samples_per_sequence,
        shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
        **kwargs,
    )
    return task_encoder.build_train_datasets(
        datasets=datasets,
        worker_config=worker_config,
        batch_size=batch_size,
        batch_drop_last=batch_drop_last,
        packing_buffer_size=packing_buffer_size,
        virtual_epoch_length=virtual_epoch_length,
        shuffle_buffer_size=shuffle_buffer_size,
    )


def get_val_dataset(
    path: Union[str, EPath, Path],
    *,
    split_part: Union[Literal["val", "test"], str] = "val",
    worker_config: WorkerConfig,
    batch_size: int,
    batch_drop_last: bool = False,
    packing_buffer_size: Optional[int] = None,
    limit: Optional[int] = None,
    task_encoder: TaskEncoder[Any, Any, Any, T] = DefaultTaskEncoder(),
    **kwargs,
) -> SavableDataset[T]:
    """
    Get the validation/test dataset with sensible defaults. See `get_dataset` for more details.

    The following recipe will be used:
      - :func:`megatron.energon.dataset_config.get_dataset_from_config` (loads the raw dataset)
      - `task_encoder.encode_sample`
      - (:class:`megatron.energon.MixDataset` if mixing)
      - :class:`megatron.energon.BatchDataset` with `task_encoder.batch` for collation
      - :class:`megatron.energon.LimitDataset` (if `limit` is set)
      - `task_encoder.encode_batch`

    Args:
        path: Path to the dataset.
        split_part: Default split part to use.
        worker_config: Worker configuration to use.
        batch_size: Size of a batch
        batch_drop_last: If true, drop the last batch if it is smaller than `batch_size`.
        limit: If set, limit the number of batches loaded from the dataset to this.
        task_encoder: Task encoder to use.
        **kwargs: Additional arguments to the dataset constructor.

    Returns:
        The loaded dataset.
    """
    loader = load_dataset(path, **_split_kwargs(kwargs))
    datasets = loader.get_datasets(
        training=False, split_part=split_part, worker_config=worker_config, **kwargs
    )
    return task_encoder.build_val_datasets(
        datasets=datasets,
        worker_config=worker_config,
        batch_size=batch_size,
        batch_drop_last=batch_drop_last,
        packing_buffer_size=packing_buffer_size,
        limit=limit,
    )


def get_val_datasets(
    path: Union[str, EPath, Path],
    *,
    split_part: Union[Literal["val", "test"], str] = "val",
    worker_config: WorkerConfig,
    batch_size: int,
    batch_drop_last: bool = False,
    packing_buffer_size: Optional[int] = None,
    limit: Optional[int] = None,
    task_encoder: TaskEncoder[Any, Any, Any, T] = DefaultTaskEncoder(),
    **kwargs,
) -> List[Tuple[SavableDataset[T], BaseCoreDataset]]:
    """
    Get the validation/test dataset with sensible defaults. See `get_dataset` for more details.

    The following recipe will be used:
      - :func:`megatron.energon.dataset_config.get_dataset_from_config` (loads the raw dataset)
      - `task_encoder.encode_sample`
      - (:class:`megatron.energon.MixDataset` if mixing)
      - :class:`megatron.energon.BatchDataset` with `task_encoder.batch` for collation
      - :class:`megatron.energon.LimitDataset` (if `limit` is set)
      - `task_encoder.encode_batch`

    Args:
        path: Path to the dataset.
        split_part: Default split part to use.
        worker_config: Worker configuration to use.
        batch_size: Size of a batch
        batch_drop_last: If true, drop the last batch if it is smaller than `batch_size`.
        limit: If set, limit the number of batches loaded from the dataset to this.
        task_encoder: Task encoder to use.
        **kwargs: Additional arguments to the dataset constructor.

    Returns:
        The loaded val datasets, with the source datasets.
    """
    loader = load_dataset(path, **_split_kwargs(kwargs))
    datasets = loader.get_datasets(
        training=False, split_part=split_part, worker_config=worker_config, **kwargs
    )
    return [
        (
            task_encoder.build_val_datasets(
                datasets=[(dataset, 1)],
                worker_config=worker_config,
                batch_size=batch_size,
                batch_drop_last=batch_drop_last,
                packing_buffer_size=packing_buffer_size,
                limit=limit,
            ),
            dataset,
        )
        for dataset, weight in datasets
    ]
