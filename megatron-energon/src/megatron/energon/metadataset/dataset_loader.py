# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from megatron.energon.dataset_config import get_dataset_from_config
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseCoreDataset
from megatron.energon.metadataset.loader_interface import DatasetLoaderInterface
from megatron.energon.worker import WorkerConfig


@dataclass
class DatasetLoader(DatasetLoaderInterface):
    """Loads a dataset from a path."""

    path: Union[str, EPath]
    split_part: Optional[str] = None
    subflavor: Optional[str] = None
    subflavors: Optional[Dict[str, Any]] = None
    shuffle_over_epochs_multiplier: int = 1
    dataset_config: str = "dataset.yaml"
    split_config: str = "split.yaml"

    weight: float = 1.0

    def get_dataset(
        self,
        *,
        training: bool,
        split_part: Optional[str] = None,
        worker_config: Optional[WorkerConfig] = None,
        subflavor: Optional[str] = None,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs: int = 1,
        **kwargs,
    ) -> BaseCoreDataset:
        """
        Args:
            training: If true, apply training randomization.
            split_part: Default split part to use.
            worker_config: Worker configuration.
            shuffle_buffer_size: Size of the sample shuffle buffer (before task encoding).
            subflavor: Subflavor to use, might be overridden by inner datasets.
            subflavors: Subflavors to use, might be overridden by inner datasets.
            shuffle_over_epochs: Shuffle the dataset over this many epochs.
            **kwargs: Additional arguments to the dataset constructor.

        Returns:
            The loaded dataset
        """
        if self.split_part is not None:
            split_part = self.split_part
        if split_part is None:
            raise ValueError("Missing split part")
        if subflavor is None:
            subflavor = self.subflavor
        if self.subflavors is not None:
            subflavors = {**self.subflavors, **(subflavors or {})}
        return get_dataset_from_config(
            self.path,
            training=training,
            split_part=split_part,
            worker_config=worker_config,
            subflavor=subflavor,
            subflavors=subflavors,
            dataset_config=self.dataset_config,
            split_config=self.split_config,
            shuffle_over_epochs=shuffle_over_epochs,
            **kwargs,
        )

    def get_datasets(
        self,
        *,
        training: bool,
        split_part: Union[Literal["train", "val", "test"], str],
        worker_config: Optional[WorkerConfig] = None,
        subflavor: Optional[str] = None,
        subflavors: Optional[Dict[str, Any]] = None,
        shuffle_over_epochs_multiplier: int = 1,
        **kwargs,
    ) -> List[Tuple[BaseCoreDataset, float]]:
        return [
            (
                self.get_dataset(
                    training=training,
                    split_part=split_part,
                    worker_config=worker_config,
                    subflavor=subflavor,
                    subflavors=subflavors,
                    shuffle_over_epochs=shuffle_over_epochs_multiplier,
                    **kwargs,
                ),
                1.0,
            )
        ]
