# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from megatron.energon.flavors.base_dataset import BaseCoreDataset
from megatron.energon.worker import WorkerConfig


class DatasetLoaderInterface(ABC):
    """General interface for a dataset loader."""

    @abstractmethod
    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: WorkerConfig,
        subflavor: Optional[str] = None,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: int = 1,
        **kwargs,
    ) -> List[Tuple[BaseCoreDataset, float]]:
        """
        Calls :func:`megatron.energon.dataset_config.get_dataset_from_config` (loads the raw dataset)
        for all innermost datasets and resolves their relative weights to absolute weights.

        Args:
            training: If true, apply training randomization.
            split_part: Default split part to use.
            worker_config: Worker configuration to use.
            subflavor: Set the default subflavor for all datasets.
            subflavors: Set the default subflavors for all datasets.
            shuffle_over_epochs_multiplier: Multiply the inner datasets
                `shuffle_over_epochs(_multiplier)` by this factor. E.g. if the inner dataset
                has `shuffle_over_epochs_multiplier=2` and this function has
                `shuffle_over_epochs_multiplier=3`, the inner dataset will be shuffled
                over 6 epochs. Shuffling over `n` epochs guarantees that each sample is seen
                exactly `n` times in `n` epochs of the inner dataset. Use -1 for shuffling over
                an infinite number of epochs (effectively, this will draw shard slices with
                replacement).
            **kwargs: Additional arguments to the dataset constructor.

        Returns:
            The instantiated core datasets and their weights.
        """
        ...
