<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Simple Library Usage

You can use Megatron Energon to load datasets in the energon dataset format. This format is essentially [WebDataset](https://github.com/webdataset/webdataset) with some meta data added.

For a moment let's assume you already have prepared a dataset in the needed format, and it's stored on
disk at `/my/dataset/path`.

```{admonition} Good to know
:class: tip
You can also store your dataset inside an S3-compatible object store and load it from there! See [Remote Dataset](remote_dataset.md)
```

You can then load the dataset like this:
```python
from megatron.energon import get_train_dataset

ds = get_train_dataset(
    '/my/dataset/path',
    batch_size=1,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
)

for batch in ds:
    # Do something with batch
    # Infer, gradient step, ...
    pass
```

At first, we call {py:meth}`get_train_dataset <megatron.energon.get_train_dataset>` (click to see signature).
The method will check what kind of dataset is on disk and instantiate the correct class for it.


This method will return a {py:class}`torch.data.IterableDataset` which we can iterate in a for-loop to get batches.
The batches will contain samples of the sample type specified in the [task encoder](task_encoders.md).

```{admonition} Good to know
:class: tip
Since we did not specify a task encoder above, the {py:class}`DefaultTaskEncoder <megatron.energon.DefaultTaskEncoder>` will be used.
It will not transform the data. For batching it will use common sense magic to pad and stack tensors or build lists if the type is unknown.
```

_Wait. Why does the dataset create batches? Shouldn't the dataloader do that?_

Energon will create batches at dataset level.
This way, each batch will contain samples from only one dataset, later you can [blend](blending_datasets) batches
from different datasets together before feeding those into the dataloader.
Check out the [](basics_flow) section to see the steps in which the data is processed.

_Why must `shuffle_buffer_size` and `max_samples_per_sequence` be set explicitly?_

As the library is designed to work on (sequential) webdatasets but still wants to provide proper shuffling, these parameters are required. To make sure, the user does not forget to set these, we enforce them to be set explicitly.
A value of 100 for both settings for image datasets seems to work well (i.e. balanced shuffling randomness vs seeking performance impact), but datasets where the samples are lots larger or smaller might require different settings.
Setting the sequence length to a very small size compared to the number of samples in the dataset will result in more random access, thus slowing down dataloading, so the recommendation is to set it to a high enough value.
At the same time, a high value reduces the shuffling randomness, which requires a larger shuffle buffer size to compensate for that (i.e. higher memory footprint and longer state restore times).

## Tutorial 1

Let's be a bit more concrete and try out the above code with a real dataset.
We are going to print the first batch and stop.

```python
from megatron.energon import get_train_dataset

ds = get_train_dataset(
    '/path/to/your/dataset',
    batch_size=1,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
)

for batch in ds:
    print(batch)
    break
```

This prints

```text
CaptioningSample(__key__=['part_00123/00403.tar/004030195'], image=tensor([[[[0.4549, 0.4784, 0.5059,  ..., 0.8392, 0.8275, 0.7961],
          [0.4549, 0.4784, 0.5020,  ..., 0.6431, 0.6275, 0.5882],
          [0.4510, 0.4706, 0.4941,  ..., 0.6235, 0.6353, 0.6078],
          ...,
          [0.4471, 0.4196, 0.4510,  ..., 0.8471, 0.8039, 0.8275],
          [0.4667, 0.4353, 0.4667,  ..., 0.8196, 0.7804, 0.8078],
          [0.4824, 0.4549, 0.4824,  ..., 0.8196, 0.7843, 0.8118]],

         [[0.3608, 0.3843, 0.4118,  ..., 0.7373, 0.7255, 0.6941],
          [0.3608, 0.3843, 0.4078,  ..., 0.5412, 0.5255, 0.4863],
          [0.3569, 0.3765, 0.4000,  ..., 0.5098, 0.5216, 0.4941],
          ...,
          [0.3608, 0.3333, 0.3647,  ..., 0.7529, 0.7098, 0.7333],
          [0.3804, 0.3490, 0.3804,  ..., 0.7255, 0.6863, 0.7137],
          [0.3961, 0.3686, 0.3961,  ..., 0.7255, 0.6902, 0.7176]],

         [[0.2510, 0.2745, 0.3020,  ..., 0.6000, 0.5882, 0.5569],
          [0.2510, 0.2745, 0.2980,  ..., 0.4039, 0.3882, 0.3490],
          [0.2471, 0.2667, 0.2902,  ..., 0.3765, 0.3882, 0.3608],
          ...,
          [0.2667, 0.2392, 0.2706,  ..., 0.6510, 0.6000, 0.6235],
          [0.2863, 0.2549, 0.2863,  ..., 0.6235, 0.5765, 0.6039],
          [0.3020, 0.2745, 0.3020,  ..., 0.6235, 0.5882, 0.6157]]]]), caption=['Cello Renting vs. Buying: Which is Right for You?'])
```

Awesome, it returns a {py:class}`CaptioningSample <megatron.energon.CaptioningSample>` with the attributes

- `__key__`: `part_00123/00403.tar/004030195`, the identifier of the sample like `TAR_FILE/INDEX`
  - All sample types will have a key. It's in the base class {py:class}`Sample <megatron.energon.Sample>` 
- `image`: The image as a tensor of shape `(1, 3, 267, 400)` (RGB image in a batch of size 1)
- `caption`: A list of strings (here just one since batch size is one) 

## Tutorial 2

Actually, we would like to use a `batch_size` of more than one, let's go with 2 for now.

```python
from megatron.energon import get_train_dataset

ds = get_train_dataset(
    '/path/to/your/dataset',
    batch_size=2,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
)

for batch in ds:
    print(batch)
    break
```

The output will be similar to above but with different shapes and lengths:

- `batch.__key__`: A list of two keys
- `batch.image`: Tensor of shape `(2, 3, 267, 400)`
- `batch.caption`: A list of two caption strings

The default [task encoder](task_encoders) automagically padded and stacked the items to a batch.
This may be ok for some cases, but usually you will want to process and batch your data differently.

Hence, we can

- either use an existing task encoder
- or define a custom one (see [](task_encoders))

## Tutorial 3

A typical usecase is to mix multiple datasets of the same (or similar type) together.
For example, you may want to mix the COCO dataset with the COYO dataset.
The easiest way to do this, is to use the metadataset pattern:

`coyo-coco-dataset.yaml`:
```yaml
__module__: megatron.energon
__class__: Metadataset
splits:
  # Train dataset, the datasets will be mixed according to their weights 
  train:
    datasets:
      - weight: 5
        path: ./coco
      - weight: 2
        path: ./coyo
  # For val and test, datasets will be concatenated
  val:
    datasets:
      - path: ./coco
  test:
    datasets:
      - path: ./coyo
```

Usage in your loader, simply use {py:func}`get_train_dataset <megatron.energon.get_train_dataset>`:
```python
from megatron.energon import get_train_dataset

ds = get_train_dataset(
    'coyo-coco-dataset.yaml',
    batch_size=4,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
)

for batch in ds:
    print(batch)
    break

```

If you need to handle samples from different datasets differently in your pipeline, you will want to use `subflavors`.
For these and other details, check out the [](metadatasets) section.

## Tutorial 4

For multi-GPU support, properly set the worker config. Either by passing in the worker config, or by
setting the global worker config. Defaults to 4 workers, if not set explicitly.

```python
from megatron.energon import get_train_dataset, get_loader, WorkerConfig

# Or set the fields `rank` and `world_size` manually
worker_config = WorkerConfig.default_worker_config(num_workers=4)

ds = get_train_dataset(
    'coyo-coco-dataset.yaml',
    batch_size=4,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
    worker_config=worker_config,
)

# Or use default pytorch loader with the same num_workers
loader = get_loader(ds, worker_config=worker_config)

for batch in loader:
    print(batch)
    break

```

## Tutorial 5

For saving and restoring the state (e.g. for autoresume), the loader must be instantiated with the savable loader.

```python
from megatron.energon import get_train_dataset, get_savable_loader, WorkerConfig

# Or set the fields `rank` and `world_size` manually
worker_config = WorkerConfig.default_worker_config(num_workers=4)

ds = get_train_dataset(
    'coyo-coco-dataset.yaml',
    batch_size=4,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
    worker_config=worker_config,
)

# Must use the savable loader, cannot use default torch loader. This provides methods to save
# and load the state of the data loader
loader = get_savable_loader(ds, worker_config=worker_config)

for i, batch in zip(range(10), loader):
    print(batch)
    break

# Save the state
state = loader.save_state()
# Could save the state now using torch.save()

# ... when loading:
# Could load the state with torch.load()

# Restore the state for a new loader
ds = get_train_dataset(
    'coyo-coco-dataset.yaml',
    batch_size=4,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
    worker_config=worker_config,
)
loader = get_savable_loader(ds, worker_config=worker_config)
loader.restore_state(state)


```
