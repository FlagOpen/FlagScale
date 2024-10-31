# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Any, Callable, Dict, Generator, Generic, Iterator, List, Tuple, TypeVar, Union

import torch

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import BaseSingleWrapperDataset
from megatron.energon.wrappers.batch_dataset import BatchDataset
from megatron.energon.wrappers.blend_dataset import BlendDataset

T_batch_in = TypeVar("T_batch_in")
T_batch = TypeVar("T_batch")


def generic_concat(batch: List[Any]) -> Any:
    """Based on the types/shapes of the batch: Will either pad and stack, or return as list.
    Recurses structures (dict, dataclass, namedtuple) and applies the same logic to each field."""
    if isinstance(batch[0], torch.Tensor):
        return concat_pad(batch)
    elif isinstance(batch[0], dict):
        return {key: generic_concat([sample[key] for sample in batch]) for key in batch[0].keys()}
    elif dataclasses.is_dataclass(batch[0]):
        return type(batch[0])(
            **{
                field.name: generic_concat([getattr(sample, field.name) for sample in batch])
                for field in dataclasses.fields(batch[0])
            }
        )
    elif isinstance(batch[0], tuple) and hasattr(batch[0], "_fields"):
        # NamedTuple
        return type(batch[0])(
            **{
                field: generic_concat([getattr(sample, field) for sample in batch])
                for field in batch[0]._fields
            }
        )
    else:
        return batch


def concat_pad(batch: List[Any]) -> Any:
    """Concat a batch of arbitrary-sized tensors padded with 0s."""
    total_bs = sum(b.shape[0] for b in batch)
    max_size = [max(b.shape[dim] for b in batch) for dim in range(1, batch[0].ndim)]
    concat_tensor = batch[0].new_zeros((total_bs, *max_size))
    b_idx = 0
    for b in batch:
        concat_tensor[(slice(b_idx, b_idx + b.shape[0]), *(slice(0, s) for s in b.shape[1:]))] = b
        b_idx += b.shape[0]
    # Pad all tensors to max_size
    return concat_tensor


def homogeneous_concat_mix(samples: List[T_batch_in]) -> T_batch:
    """
    Mixes a list of batches into a single batch. The default implementation is to concat the
    batches if they are all of the same type, otherwise return a list of batches.

    Args:
        samples: THe samples to mix.

    Returns:
        The mixed batch.
    """
    first_type = type(samples[0])
    assert all(first_type == type(sample) for sample in samples)
    # All the same type -> concat batches
    return generic_concat(samples)


class MixBatchDataset(BaseSingleWrapperDataset[T_batch_in, T_batch], Generic[T_batch_in, T_batch]):
    """
    This dataset wrapper blends multiple iterable datasets together give a weight.
    The datasets may be infinite. This dataset is always infinite.
    Effectively combines :class:`megatron.energon.BlendDataset` and :class:`megatron.energon.BatchDataset`.
    """

    def __init__(
        self,
        *dataset_weights: Tuple[SavableDataset[T_batch_in], float],
        batch_size: int,
        batch_mix_fn: Callable[
            [List[T_batch_in]], Union[T_batch, Generator[T_batch, None, None]]
        ] = lambda x: x,
        worker_config: WorkerConfig,
    ):
        """Construct a BlendDataset.

        Args:
            dataset_weights: Each argument should be a tuple of (dataset, weight) with a weight
                between 0 and 1. The output samples are sampled from the input datasets with the
                given probabilities. The datasets should have a batch size of 1, otherwise the
                whole batches will be sampled.
            batch_size: The batch size to output.
            batch_mix_fn: A function that takes a list of samples from the input datasets and
                returns a batch sample. The default implementation returns a list of batches.
                For homogeneous datasets, it is recommended to use the
                :func:`megatron.energon.homogeneous_concat_mix` which concatenates the batches. May raise
                :exc:`megatron.energon.SkipSample` to skip a sample. May also return a generator, which
                will be iterated over to produce batches.
            worker_config: Configuration for the workers.
        """
        super().__init__(
            BatchDataset(
                BlendDataset(*dataset_weights, worker_config=worker_config),
                batch_size=batch_size,
                batcher=batch_mix_fn,
                worker_config=worker_config,
            )
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_batch]:
        yield from self.dataset

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "dataset": self.dataset.config(),
        }

    def __str__(self):
        return f"MixBatchDataset(dataset={self.dataset})"
