# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, TypeVar

from torch.utils.data import DataLoader

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.savable_loader import BasicDataLoader, SavableDataLoader
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.gc_dataset import GcDataset

T = TypeVar("T")


def get_savable_loader(
    dataset: SavableDataset[T],
    *,
    worker_config: WorkerConfig,
    checkpoint_every_sec: float = 60,
    checkpoint_every_min_n_samples: Optional[int] = None,
    n_checkpoints: int = 2,
) -> SavableDataLoader[T]:
    """

    Get a dataloader for the given dataset.

    Args:
        dataset: The dataset to create a loader for.
        worker_config: The worker config. Set the same value to the `get(_val|_train)_dataset` call.
        checkpoint_every_sec: This is the time in seconds after which an internal checkpoint is
            saved. It may take the same duration to restore a checkpoint, but introduces additional
            overhead during reading data from the dataset, so this should be chosen accordingly.
            Only applies if using workers.
        checkpoint_every_min_n_samples: Overwrites the minimum number of samples between
            checkpoints. Defaults to `number of workers * 2`. Only applies if using workers.
        n_checkpoints: The number of internal checkpoints to keep. Only applies if using workers.

    Returns:
        The instantiated :class:`megatron.energon.SavableDataLoader`, yielding batches from the dataset,
        allowing to save the state of the dataset.
    """
    dataset.verify_worker_config(worker_config)
    return SavableDataLoader(
        dataset,
        worker_config=worker_config,
        checkpoint_every_sec=checkpoint_every_sec,
        checkpoint_every_min_n_samples=checkpoint_every_min_n_samples,
        n_checkpoints=n_checkpoints,
    )


def get_loader(
    dataset: SavableDataset[T],
    *,
    worker_config: WorkerConfig,
) -> BasicDataLoader[T]:
    """
    Get a dataloader for the given dataset.

    Args:
        dataset: The dataset to create a loader for.
        worker_config: The worker config. Set the same value to the `get(_val|_train)_dataset` call.

    Returns:
        The instantiated :class:`torch.data.DataLoader`, yielding batches from the dataset.
    """
    dataset.verify_worker_config(worker_config)
    return BasicDataLoader(
        GcDataset(dataset),
        worker_config=worker_config,
    )
