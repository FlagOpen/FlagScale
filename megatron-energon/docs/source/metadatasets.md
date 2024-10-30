<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Metadataset

Metadatasets allow combining datasets together in a variety of ways by using a yaml config file.
This is useful for example if you want to mix multiple datasets together, and especially if you want to reuse that combination.

Simple format of the `metadataset.yaml` config files:
```yaml
# Outer object must be a `Metadataset`
__module__: megatron.energon
__class__: Metadataset
splits:
  # Split defines the train/val/test subsets composition 
  train:
    # This can be any `MetadatasetBase` class. E.g. a `DatasetMetadataset`, or a `MixMetadataset`.
    # Mix the following datasets
    datasets:
      - weight: 5
        path: ./coco
      - weight: 2
        path: ./coyo
      - weight: 1
        path: ./coyo
  val:
    # Only use coco-val for val
    datasets:
      - path: ./coco
  test:
    # Only use coyo-test for test
    datasets:
      - path: ./coyo
```


Usage in your loader, simply use {py:func}`get_train_dataset <megatron.energon.get_train_dataset>`:
```python
from megatron.energon import get_train_dataset

ds = get_train_dataset(
    'demo-metadataset.yaml',
    batch_size=4,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
)

for batch in ds:
    print(batch)
    break

```

Advanced format of the `metadataset.yaml` config files:
```yaml
# Outer object must be a `Metadataset`
__module__: megatron.energon
__class__: Metadataset
splits:
  # Split defines the train/val/test subsets composition 
  train:
    # This can be any `MetadatasetBase` class. E.g. a `DatasetMetadataset`, or a `MixMetadataset`.
    # Mix the following datasets
    datasets:
      - weight: 5
        path: ./coco
        # Set the __subflavor__ property of the coco samples
        subflavor: small_images
        # Set the __subflavors__ property of the samples
        subflavors:
          augmentation_type: small_images
          text_length: short
      # Combine coyo-train and coyo-val
      - weight: 2
        path: ./coyo
        split_part: train
        # Set the __subflavors__ property of the samples
        subflavors:
          augmentation_type: large_images
          text_length: short
      - weight: 1
        path: ./coyo
        split_part: val
        # Set the __subflavors__ property of the samples
        subflavors:
          augmentation_type: large_images
          text_length: short
  # For val and test, MixDataset will actually concatenate the datasets
  val:
    # Only use coco val for val
    datasets:
      - path: ./coco
        subflavor: small_images
        subflavors:
          augmentation_type: small_images
          text_length: short
  test:
    datasets:
      - path: ./coyo
```

## Customizing mixing

The Task-Encoder allows to customize the mixing of datasets given their accumulated mixing weights:

```py

# All the typing is optional
class CaptioningTaskEncoder(
    DefaultTaskEncoder[CaptioningSample, CaptioningSample, CaptioningRawBatch, CaptioningBatch]
):
    ...
    
    def build_train_datasets(
        self,
        *,
        datasets: List[Tuple[BaseCoreDataset[CaptioningSample], float]],
        worker_config: WorkerConfig,
        batch_size: int,
        batch_drop_last: bool,
        virtual_epoch_length: int = 0,
    ) -> SavableDataset[ImageTaskBatch]:
        # The default implementation uses MixDataset, which mixes the datasets according to their weights
        # This could be customized, e.g. to batch the datasets first (i.e. each batch only contains data from a single datset)
        # and then blend, which would yield the same distribution.
        dataset = BlendDataset(
            *datasets,
            worker_config=worker_config,
        )
        # Build batches from mixed samples
        dataset = self.build_batch(
            dataset,
            batch_size=batch_size,
            batch_drop_last=batch_drop_last,
            worker_config=worker_config,
        )
        # Optionally epochize
        if virtual_epoch_length > 0:
            dataset = EpochizeDataset(
                dataset,
                length=virtual_epoch_length,
                worker_config=worker_config,
            )
        return dataset

```


## Classes
* {py:class}`DatasetLoaderInterface <megatron.energon.DatasetLoaderInterface>`: Common interface for dataset loaders. Provides methods for constructing/loading the actual train- or val-mode dataset.
  * {py:class}`Metadataset <megatron.energon.Metadataset>`: The metadataset loader using the yaml example above. Mixes datasets for train-mode, and concatenates for val-mode.
  * {py:class}`DatasetLoader <megatron.energon.DatasetLoader>`: The dataset loader using a dataprepped folder (containing `.nv-meta` folder).

## Functions
* {py:func}`get_train_dataset <megatron.energon.get_train_dataset>`: Returns the train-mode (meta)dataset.
* {py:func}`get_val_dataset <megatron.energon.get_val_dataset>`: Returns the val-mode (meta)dataset.
