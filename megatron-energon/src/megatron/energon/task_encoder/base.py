# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
import inspect
from abc import ABC
from dataclasses import is_dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Hashable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch

from megatron.energon.flavors.base_dataset import (
    BaseCoreDataset,
    PinMemoryMixin,
    Sample,
    SavableDataset,
)
from megatron.energon.flavors.crude import CrudeSample, CrudeWebdataset
from megatron.energon.rng import SystemRng
from megatron.energon.task_encoder.cooking import Cooker
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers import (
    BatchDataset,
    BlendDataset,
    ConcatDataset,
    EpochizeDataset,
    GroupBatchDataset,
    LimitDataset,
    LogSampleDataset,
    MapDataset,
    PackingDataset,
    ShuffleBufferDataset,
)

T = TypeVar("T")
V = TypeVar("V")
T_sample = TypeVar("T_sample")
T_encoded_sample = TypeVar("T_encoded_sample")
T_raw_batch = TypeVar("T_raw_batch")
T_batch = TypeVar("T_batch")


FeatureBatcher = Callable[[List[Any]], Any]


def generic_batch(batch: List[Any]) -> Any:
    """Based on the types/shapes of the batch: Will either pad and stack, or return as list.
    Recurses structures (dict, dataclass, namedtuple) and applies the same logic to each field."""
    if isinstance(batch[0], torch.Tensor):
        return batch_pad_stack(batch)
    elif isinstance(batch[0], dict):
        return {key: generic_batch([sample[key] for sample in batch]) for key in batch[0].keys()}
    elif is_dataclass(batch[0]):
        return type(batch[0])(
            **{
                field.name: generic_batch([getattr(sample, field.name) for sample in batch])
                for field in dataclasses.fields(batch[0])
            }
        )
    elif isinstance(batch[0], tuple) and hasattr(batch[0], "_fields"):
        # NamedTuple
        return type(batch[0])(
            **{
                field: generic_batch([getattr(sample, field) for sample in batch])
                for field in batch[0]._fields
            }
        )
    else:
        return batch_list(batch)


def batch_stack(batch: List[Any]) -> Any:
    """Stack a batch of tensors."""
    return torch.stack(batch, dim=0)


def batch_pad_stack(batch: List[Any]) -> Any:
    """Stack a batch of arbitrary-sized tensors padded with 0s."""
    max_size = [max(b.shape[dim] for b in batch) for dim in range(batch[0].ndim)]
    batch_tensor = batch[0].new_zeros((len(batch), *max_size))
    for i, b in enumerate(batch):
        batch_tensor[(i, *(slice(0, s) for s in b.shape))] = b
    # Pad all tensors to max_size
    return batch_tensor


def batch_list(batch: List[Any]) -> Any:
    """Stack a batch of tensors padded with 0s."""
    return batch


def stateless(
    fn: Optional[Callable[..., T_sample]] = None, *, restore_seeds: bool = False
) -> Callable[..., T_sample]:
    """Decorator to mark a function of the task encoder as restorable.

    Args:
        fn: The function to decorate.
        restore_seeds: Whether to restore the seeds for the function. I.e. the seeds are set
            from the sample index and the worker seed, such that they can be restored when a sample
            is restored from that function.

    Usage:

        @stateless
        def encode_sample(self, sample: T_sample) -> T_encoded_sample:
            ...


        # Or if randomness is used (e.g. for augmentations):
        @stateless(restore_seeds=True)
        def encode_sample(self, sample: T_sample) -> T_encoded_sample:
            ...
    """
    if fn is None:
        return lambda f: stateless(f, restore_seeds=restore_seeds)
    if restore_seeds:
        worker_seed = None

        @functools.wraps(fn)
        def seed_wrapper_generator(self, *args, **kwargs):
            nonlocal worker_seed
            if worker_seed is None:
                worker_seed = WorkerConfig.active_worker_config.worker_seed()

            # Save the RNG states and set the new seed
            outer_rng_state = SystemRng.save_state()

            # Before constructing the generator and before the first
            # iteration, set inner RNG based on seed computed
            # from worker_seed and current sample index
            SystemRng.seed_args(worker_seed, self.current_sample_index)

            it = iter(fn(self, *args, **kwargs))

            inner_rand_state = None

            while True:

                if inner_rand_state is not None:
                    # Restore inner random state before calling the generator
                    # This will not be done on the first iteration
                    SystemRng.restore_state(inner_rand_state)

                try:
                    # Now call the generator. This will yield the sample
                    # But note it may also throw an exception or a StopIteration
                    sample = next(it)

                    # Save inner random state after calling the generator
                    inner_rand_state = SystemRng.save_state()
                except StopIteration:
                    # We're stopping here, but the outer random state
                    # will be restored before returning (in finally below)
                    break
                finally:
                    # Restore outer rand state before yielding or when an exception was raised
                    SystemRng.restore_state(outer_rng_state)

                # Now yield the sample.
                # This will give control back to the caller who may
                # change the random state.
                yield sample

                # Save outer random state after yielding
                outer_rng_state = SystemRng.save_state()

        @functools.wraps(fn)
        def seed_wrapper(self, *args, **kwargs):
            nonlocal worker_seed
            if worker_seed is None:
                worker_seed = WorkerConfig.active_worker_config.worker_seed()

            # Save the RNG states and set the new seed
            rng_state = SystemRng.save_state()

            SystemRng.seed_args(worker_seed, self.current_sample_index)

            try:
                return fn(self, *args, **kwargs)
            finally:
                # Restore the RNGs
                SystemRng.restore_state(rng_state)

        if inspect.isgeneratorfunction(fn):
            setattr(seed_wrapper_generator, "__stateless__", True)
            return seed_wrapper_generator
        else:
            setattr(seed_wrapper, "__stateless__", True)
            return seed_wrapper

    setattr(fn, "__stateless__", True)
    return fn


@dataclasses.dataclass
class Batch(PinMemoryMixin):
    """Base class for a batch dataclass. Provides a default implementation for pinning memory."""


class TaskEncoder(ABC, Generic[T_sample, T_encoded_sample, T_raw_batch, T_batch]):
    """
    Base class for task encoders.

    Task encoding follows these steps:
      0. Data comes from the dataset
      1. :meth:`megatron.energon.TaskEncoder.encode_sample` is called on each sample
      2. :meth:`megatron.energon.TaskEncoder.batch` is called on the list of encoded samples
      3. :meth:`megatron.energon.TaskEncoder.encode_batch` is called on the batch
      4. :meth:`megatron.energon.TaskEncoder.to_device` is called on the encoded batch
      5. resulting encoded batch is passed to the network
    """

    cookers: Sequence[Cooker] = ()

    @stateless
    def cook_crude_sample(self, sample: Union[T_sample, CrudeSample]) -> T_sample:
        if isinstance(sample, CrudeSample):
            for cooker in self.cookers:
                if cooker.is_match(sample):
                    return cooker.cook(sample)

            raise NotImplementedError(
                "You are using crude samples but not providing a way to cook them."
            )
        else:
            assert isinstance(sample, Sample), "Sample must be a complete Sample or a CrudeSample"
            return sample

    @stateless
    def encode_sample(
        self, sample: T_sample
    ) -> Union[T_encoded_sample, Generator[T_encoded_sample, None, None]]:
        """Encode a single sample. May raise :exc:`megatron.energon.SkipSample` to skip a sample.
        Alternatively, this can be a generator that yields (or ignores) new samples."""
        return sample

    @stateless
    def batch(self, samples: List[T_encoded_sample]) -> T_raw_batch:
        """Move a batch to a device. May raise :exc:`megatron.energon.SkipSample` to skip a batch."""
        return self._batch(samples, type(samples[0]))

    def batch_group_criterion(self, sample: T_encoded_sample) -> Hashable:
        """Return a group criterion for the sample. Default implementation does not group
        (effectively, it returns a single value (`None`), thus only one group is used). May raise
        :exc:`megatron.energon.SkipSample` to skip a batch."""
        return None

    @stateless
    def encode_batch(self, batch: T_raw_batch) -> Union[T_batch, Generator[T_batch, None, None]]:
        """Encode a batch of samples. May raise :exc:`megatron.energon.SkipSample` to skip a batch.
        Alternatively, this can be a generator that yields (or ignores) new batches."""
        return batch

    def _batch(
        self,
        samples: List[T_sample],
        result_type: Type[T_raw_batch],
        actions: Optional[Dict[str, FeatureBatcher]] = None,
        default_action: FeatureBatcher = generic_batch,
    ) -> T_raw_batch:
        """
        Batch a list of samples.

        Args:
            samples: The samples to batch
            result_type: Type of the result (might be dict, dataclass, or namedtuple)
            actions: For each field (=key), may specify a specific batcher
            default_action: The batcher to apply to all fields not in `action`

        Returns:
            The batched result
        """
        # Get dict of samples
        if isinstance(samples[0], dict):
            list_samples = {key: [sample[key] for sample in samples] for key in samples[0].keys()}
        elif is_dataclass(samples[0]):
            list_samples = {
                field.name: [getattr(sample, field.name) for sample in samples]
                for field in dataclasses.fields(samples[0])
            }
        elif isinstance(samples[0], tuple) and hasattr(samples[0], "_fields"):
            # NamedTuple
            list_samples = {
                field: [getattr(sample, field) for sample in samples]
                for field in samples[0]._fields
            }
        else:
            raise ValueError("Unrecognized sample type.")
        # Convert each field
        if actions is not None:
            list_samples = {
                key: default_action(value) if key not in actions else actions[key](value)
                for key, value in list_samples.items()
            }
        else:
            list_samples = {key: default_action(value) for key, value in list_samples.items()}
        # Construct result
        if issubclass(result_type, dict):
            return list_samples
        elif dataclasses.is_dataclass(result_type) or issubclass(result_type, tuple):
            # DataClass or NamedTuple
            return result_type(**list_samples)
        else:
            raise ValueError("Unrecognized result type.")

    def select_samples_to_pack(self, samples: List[T_sample]) -> List[List[T_sample]]:
        """
        For packing, selects the samples to be packed together.
        Packing is only active when packing_buffer_size is set.
        Internally this stage is called "pre_packing".

        Args:
            samples: The samples to pre-pack. A full buffer will be passed into the function.

        Returns: The pre-packed samples as a list of lists of samples.
        """
        raise NotImplementedError("Packing only effective when overridden.")

    def pack_selected_samples(self, samples: List[T_sample]) -> T_sample:
        """
        Given one set of samples to pack, returns the final packed sample.
        Packing is only active when packing_buffer_size is set.
        Internally this stage is called "final_packing".

        Args:
            samples: The samples to pack into a single sample

        Returns: The final packed sample.
        """
        raise NotImplementedError("Packing only effective when overridden.")

    def build_batch(
        self,
        dataset: SavableDataset[T_encoded_sample],
        batch_size: Optional[int],
        batch_drop_last: bool = False,
        packing_buffer_size: Optional[int] = None,
        worker_config: Optional[WorkerConfig] = None,
    ) -> SavableDataset[T_raw_batch]:
        """Applies the batcher to the dataset."""

        if (
            getattr(self.batch_group_criterion, "__func__", None)
            is not TaskEncoder.batch_group_criterion
        ):
            assert batch_size is not None, "batch_size must be set if batch_group_criterion is set"
            assert packing_buffer_size is None, "Packing not supported when grouping"
            dataset = GroupBatchDataset(
                dataset,
                batch_size=batch_size,
                group_criterion=self.batch_group_criterion,
                batcher=self.batch,
                drop_last=batch_drop_last,
                worker_config=worker_config,
            )
        else:
            # No grouping is active

            if packing_buffer_size is not None:
                select_samples_to_pack_provided = (
                    getattr(self.select_samples_to_pack, "__func__", None)
                    is not TaskEncoder.select_samples_to_pack
                )
                pack_selected_samples_provided = (
                    getattr(self.pack_selected_samples, "__func__", None)
                    is not TaskEncoder.pack_selected_samples
                )

                assert (
                    select_samples_to_pack_provided and pack_selected_samples_provided
                ), "Both select_samples_to_pack and pack_selected_samples methods must be provided in the TaskEncoder when using packing_buffer_size"

                dataset = PackingDataset(
                    dataset,
                    buffer_size=packing_buffer_size,
                    pre_packer=self.select_samples_to_pack,
                    final_packer=self.pack_selected_samples,
                    final_packer_stateless=getattr(
                        self.pack_selected_samples, "__stateless__", False
                    ),
                    worker_config=worker_config,
                )

            if batch_size is not None:
                dataset = BatchDataset(
                    dataset,
                    batch_size=batch_size,
                    batcher=self.batch,
                    batcher_stateless=getattr(self.batch, "__stateless__", False),
                    drop_last=batch_drop_last,
                    worker_config=worker_config,
                )

                if getattr(self.encode_batch, "__func__", None) is not TaskEncoder.encode_batch:
                    dataset = MapDataset(
                        dataset,
                        self.encode_batch,
                        worker_config=worker_config,
                        stateless_map_fn=getattr(self.encode_batch, "__stateless__", False),
                    )
            else:
                assert (
                    getattr(self.encode_batch, "__func__", None) is TaskEncoder.encode_batch
                ), "batch_size is not set, but encode_batch is not the default."
                assert (
                    getattr(self.batch, "__func__", None) is TaskEncoder.batch
                ), "batch_size is not set, but batch is not the default."

        return dataset

    def build_cook_crude_sample(
        self,
        dataset: SavableDataset[Union[T_sample, dict]],
        worker_config: WorkerConfig,
    ) -> SavableDataset[T_sample]:
        """Applies the sample cooker to the dataset if we have cookers registered."""
        if (
            self.cookers
            or getattr(self.build_cook_crude_sample, "__func__", None)
            is not TaskEncoder.build_cook_crude_sample
        ):
            dataset = MapDataset(
                dataset,
                self.cook_crude_sample,
                worker_config=worker_config,
                stateless_map_fn=getattr(self.cook_crude_sample, "__stateless__", False),
            )
        return dataset

    def build_encode_sample(
        self,
        dataset: SavableDataset[T_sample],
        worker_config: WorkerConfig,
    ) -> SavableDataset[T_encoded_sample]:
        """Applies the sample encoder to the dataset."""
        if getattr(self.encode_sample, "__func__", None) is not TaskEncoder.encode_sample:
            dataset = MapDataset(
                dataset,
                self.encode_sample,
                worker_config=worker_config,
                stateless_map_fn=getattr(self.encode_sample, "__stateless__", False),
            )
        return dataset

    def build_train_datasets(
        self,
        *,
        datasets: List[Tuple[BaseCoreDataset[T_sample], float]],
        worker_config: WorkerConfig,
        batch_size: int,
        batch_drop_last: bool = False,
        packing_buffer_size: Optional[int] = None,
        virtual_epoch_length: int = 0,
        shuffle_buffer_size: Optional[int] = None,
    ) -> SavableDataset[T_batch]:
        """Combines train datasets to a single dataset."""

        # Check if there's a CrudeWebdataset but no cookers
        for dataset, _ in datasets:
            if isinstance(dataset, CrudeWebdataset):
                assert self.cookers, "CrudeWebdataset found, but no cookers registered."

        if len(datasets) > 1:
            dataset = BlendDataset(
                *datasets,
                worker_config=worker_config,
            )
        elif len(datasets) == 1:
            dataset = datasets[0][0]
        else:
            raise ValueError("No datasets given.")
        if shuffle_buffer_size is not None and shuffle_buffer_size > 1:
            dataset = ShuffleBufferDataset(
                dataset,
                size=shuffle_buffer_size,
                worker_config=worker_config,
            )
        dataset = self.build_cook_crude_sample(dataset, worker_config=worker_config)
        dataset = self.build_encode_sample(dataset, worker_config=worker_config)
        dataset = self.build_batch(
            dataset,
            batch_size=batch_size,
            batch_drop_last=batch_drop_last,
            packing_buffer_size=packing_buffer_size,
            worker_config=worker_config,
        )
        if virtual_epoch_length > 0:
            dataset = EpochizeDataset(
                dataset,
                length=virtual_epoch_length,
                worker_config=worker_config,
            )
        if worker_config.should_log(level=1):
            dataset = LogSampleDataset(dataset, mode="train", worker_config=worker_config)
        return dataset

    def build_val_datasets(
        self,
        *,
        datasets: List[Tuple[BaseCoreDataset[T_sample], float]],
        worker_config: WorkerConfig,
        batch_size: int,
        batch_drop_last: bool = False,
        packing_buffer_size: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> SavableDataset[T_batch]:
        """Combines val datasets to a single dataset."""

        # Check if there's a CrudeWebdataset but no cookers
        for dataset, _ in datasets:
            if isinstance(dataset, CrudeWebdataset):
                assert self.cookers, "CrudeWebdataset found, but no cookers registered."

        if len(datasets) > 1:
            dataset = ConcatDataset(
                *[dataset for dataset, _ in datasets],
                worker_config=worker_config,
            )
        elif len(datasets) == 1:
            dataset = datasets[0][0]
        else:
            raise ValueError("No datasets given.")
        dataset = self.build_cook_crude_sample(dataset, worker_config=worker_config)
        dataset = self.build_encode_sample(dataset, worker_config=worker_config)
        dataset = self.build_batch(
            dataset,
            batch_size=batch_size,
            batch_drop_last=batch_drop_last,
            packing_buffer_size=packing_buffer_size,
            worker_config=worker_config,
        )
        if limit is not None and limit > 0:
            dataset = LimitDataset(
                dataset,
                length=limit,
                worker_config=worker_config,
                reset_after_epoch=True,
            )
        if worker_config.should_log(level=2):
            dataset = LogSampleDataset(dataset, mode="val", worker_config=worker_config)
        return dataset

    @property
    def current_batch_index(self) -> int:
        """Returns the current index for the next batch yielded from the current worker. Each batch
        on the current rank will get a strictly increasing unique number. Counting happens on each
        rank separately (i.e. each rank will get the same numbers for same batch index)."""
        assert (
            WorkerConfig.active_worker_config is not None
        ), "The batch_index can only be fetched within the worker, and to be usable, you must use the get_(savable_)loader methods provided from the package."
        return WorkerConfig.active_worker_config.active_worker_batch_index

    @property
    def current_sample_index(self) -> int:
        """Returns the current index for the next sample yielded from the current routine (e.g.
        for `encode_sample`, `batch`, or `encode_batch`). Each routine will get a number
        representing the number of calls to that function. Across workers, this number will be
        unique, but it is not synced across workers, thus it may raise in different intervals (e.g.
        if batching does not work the same for all batches). When restoring a sample, this number is
        also restored and can be relied on for deterministic randomness reproduction of a sample."""
        assert (
            WorkerConfig.active_worker_config is not None
        ), "The batch_index can only be fetched within the worker, and to be usable, you must use the get_(savable_)loader methods provided from the package."
        return WorkerConfig.active_worker_config.active_worker_sample_index


class DefaultTaskEncoder(
    TaskEncoder[T_sample, T_encoded_sample, T_raw_batch, T_batch],
    ABC,
    Generic[T_sample, T_encoded_sample, T_raw_batch, T_batch],
):
    """
    The default task encoder supports automagically mapping to target types.
    You may override any methods to customize the behavior. By default, `encode_sample` is the
    identity function, `batch` calls `_batch` with the type of the first sample, and `encode_batch`
    is also the identity function. If you set any of `encoded_sample_type`, 'raw_batch_type' or
    `batch_type`, the corresponding method return that type, where it automatically maps the fields
    (by name) to your new type.
    """

    _encoded_sample_type: Optional[Type[T_encoded_sample]]
    _raw_batch_type: Optional[Type[T_raw_batch]]
    _batch_type: Optional[Type[T_batch]]

    def __init__(
        self,
        *,
        encoded_sample_type: Optional[Type[T_encoded_sample]] = None,
        raw_batch_type: Optional[Type[T_raw_batch]] = None,
        batch_type: Optional[Type[T_batch]] = None,
    ):
        """
        Initialize the default task encoder.
        Types may be:
          * A `@dataclass` class: Return that typed dataclass. Field names must match the input
            fields.
          * A `NamedTuple` class: Return that typed namedtuple. Field names must match the input
            fields.
          * `dict`: Simply return the input as dict with field names as keys.

        Args:
            encoded_sample_type: Type of encoded samples (before batching)
            raw_batch_type: Type of the batched samples (after batching)
            batch_type: Type of the encoded batched samples
        """
        self._encoded_sample_type = encoded_sample_type
        self._raw_batch_type = raw_batch_type
        self._batch_type = batch_type

    @stateless
    def encode_sample(
        self, sample: T_sample
    ) -> Union[T_encoded_sample, Generator[T_encoded_sample, None, None]]:
        """Encode a single sample. The default implementation converts to the
        _encoded_sample_type."""
        if self._encoded_sample_type is None or isinstance(sample, self._encoded_sample_type):
            return sample
        if is_dataclass(sample):
            fields = {
                field.name: getattr(sample, field.name) for field in dataclasses.fields(sample)
            }
        elif isinstance(sample, tuple) and hasattr(sample, "_fields"):
            fields = {field: getattr(sample, field) for field in sample._fields}
        elif isinstance(sample, dict):
            fields = sample
        else:
            raise ValueError("Unrecognized sample type.")
        if issubclass(self._encoded_sample_type, dict):
            return fields
        elif dataclasses.is_dataclass(self._encoded_sample_type) or issubclass(
            self._encoded_sample_type, tuple
        ):
            # DataClass or NamedTuple
            return self._encoded_sample_type(**fields)
        else:
            raise ValueError("Unrecognized encoded sample type.")

    @stateless
    def batch(self, samples: List[T_encoded_sample]) -> T_raw_batch:
        """Batch a list of samples. The default implementation uses default batching to convert
        to _batch_type."""
        actions = None
        if isinstance(samples[0], Sample):
            actions = {
                "__subflavor__": lambda x: x,
                "__subflavors__": lambda x: x,
            }
        return self._batch(
            samples,
            type(samples[0]) if self._raw_batch_type is None else self._raw_batch_type,
            actions=actions,
        )

    @stateless
    def encode_batch(self, batch: T_raw_batch) -> Union[T_batch, Generator[T_batch, None, None]]:
        """Encode a batch of samples. The default implementation converts to the
        _encoded_batch_type."""
        if self._batch_type is None or self._raw_batch_type == self._batch_type:
            return batch
        if is_dataclass(batch):
            fields = {field.name: getattr(batch, field.name) for field in dataclasses.fields(batch)}
        elif isinstance(batch, tuple) and hasattr(batch, "_fields"):
            fields = {field: getattr(batch, field) for field in batch._fields}
        elif isinstance(batch, dict):
            fields = batch
        else:
            raise ValueError("Unrecognized sample type.")
        if issubclass(self._batch_type, dict):
            return fields
        elif dataclasses.is_dataclass(self._batch_type) or issubclass(self._batch_type, tuple):
            # DataClass or NamedTuple
            return self._batch_type(**fields)
        else:
            raise ValueError("Unrecognized encoded sample type.")


class AugmentTaskEncoder(
    TaskEncoder[T_sample, T_encoded_sample, T_raw_batch, T_batch],
    Generic[T_sample, T_encoded_sample, T_raw_batch, T_batch],
):
    """Augment a task encoder with additional functionality. By default, delegates everything to the
    original task encoder."""

    def __init__(self, task_encoder: TaskEncoder[T_sample, T_encoded_sample, T_raw_batch, T_batch]):
        """Initialize the augmenting task encoder.

        Args:
            task_encoder: The delegate task encoder. All calls will by default be forwarded to this.
        """
        self._task_encoder = task_encoder

    def encode_sample(self, sample: T_sample) -> T_encoded_sample:
        return self._task_encoder.encode_sample(sample)

    def batch(self, samples: List[T_encoded_sample]) -> T_raw_batch:
        return self._task_encoder.batch(samples)

    def batch_group_criterion(self, sample: T_encoded_sample) -> Hashable:
        return self._task_encoder.batch_group_criterion(sample)

    def encode_batch(self, batch_data: T_raw_batch) -> T_batch:
        return self._task_encoder.encode_batch(batch_data)
