<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Blending datasets

```{warning}
This page shows the **internals of blending**, which are also exposed as an interface, but we recommend using [Metadatasets](metadatasets.md) instead.
```

For blending, use the {py:class}`BlendDataset <megatron.energon.BlendDataset>` class. It takes a list of tuples, where each tuple contains a dataset and a weight. The weight determines the probability of the dataset being used to sample a batch. The weights are normalized, so they don't have to sum up to 1.
The {py:class}`BlendDataset <megatron.energon.BlendDataset>` will yield a full batch of each dataset if used in this way.
For validation, we recommend using the {py:class}`ConcatDataset <megatron.energon.BlendDataset>` class instead, as you likely don't want to randomize it.

This blending will always yield full batches of each dataset, which statistically is totally fine, as it's distributed over all ranks(=GPUs), and each rank might get a different batch, thus giving a proper distribution over all ranks (if there are enough in total).
If you need to intermix batches, check out [Mixing datasets](#mixing-datasets)

For details on task encoders, see [Task Encoders](task_encoders.md).

```python
from megatron.energon import get_train_dataset, get_val_dataset, get_loader, BlendDataset, ConcatDataset

# Define your task encoders for ImageTask and CaptioningTask as in the task_encoders.md example.
# ...

# Combine two datasets (here even of different type) with weight 60/40%, i.e. 60% of batches 
# will be of type ImageBatch, 40% of batches will be CaptioningBatch
train_loader = get_loader(BlendDataset(
    (
        get_train_dataset(
            '/my/dataset/image_dataset',
            batch_size=32,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
            task_encoder=ImageTask(),
        ),
        0.6,
    ),
    (
        get_train_dataset(
            '/my/dataset/captioning_dataset',
            batch_size=32,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
            task_encoder=CaptioningTask(),
        ),
        0.4,
    ),
))

val_loader = get_loader(ConcatDataset(
    get_val_dataset('/my/dataset/image_dataset', batch_size=32, task_encoder=ImageTask()),
    get_val_dataset('/my/dataset/captioning_dataset', batch_size=32, task_encoder=CaptioningTask()),
))

# Iterating over train_loader/val_loader will yield batches of type Union[CaptioningBatch, ImageBatch],
# i.e. it will always yield a full batch of CaptioningBatch or ImageBatch. They will not be mixed
# within a single batch.
for sample in train_loader:
    # sample is either CaptioningBatch or ImageBatch
    pass

```

(mixing-datasets)=
## Mixing datasets of homogeneous type

Mixing is a bit more versatile than blending, but it allows to mix within a single batch. If needed, you will need to 
epochize manually after blending. Same holds for batch encoding.

```python
from megatron.energon import MixBatchDataset, get_train_dataset, get_loader, homogeneous_concat_mix

# Define your task encoder for CaptioningTask as in the task_encoders.md example.
# ...

# Combine two datasets (here even of different type) with weight 60/40%, i.e. batches will consist
# 60% of first dataset, and 40% of second dataset.
mix_dataset = MixBatchDataset(
    # It is important to set the inner batch_size to 1, to ensure proper mixing
    (
        get_train_dataset(
            '/my/dataset/captioning_dataset_1',
            batch_size=1,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
            task_encoder=CaptioningTask(),
        ),
        6,
    ),
    (
        get_train_dataset(
            '/my/dataset/captioning_dataset_2',
            batch_size=1,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
            task_encoder=CaptioningTask(),
        ),
        4,
    ),
    batch_size=32,
    batch_mix_fn=homogeneous_concat_mix,
)

# If needed, epochize here:
#mix_dataset = EpochizeDataset(mix_dataset, num_epochs=10)
# If needed, map the batch
#mix_dataset = MapDataset(mix_dataset, map_fn=lambda batch: batch)

train_loader = get_loader(mix_dataset)

# Iterating over train_loader will yield batches of type CaptioningBatch.
for sample in train_loader:
    # sample is CaptioningBatch
    pass

```

## Mixing datasets of heterogeneous types

Mixing heterogeneous datasets is still possible, but requires more work. You need to define a mixer, which can combine the different types of batches into a common format.

```python
from megatron.energon import get_train_dataset, get_loader, MixBatchDataset, concat_pad

# Define your task encoder for CaptioningTask as in the task_encoders.md example.
# ...

def my_mix_fn(batch: List[Union[CaptioningBatch, ImageBatch]]) -> ImageWithOptionalCaptionBatch:
    # Here, concat the images, and return a list of captions which may contain None.
    return ImageWithOptionalCaptionBatch(
        images=concat_pad([entry.image for entry in batch]),
        captions=[entry.caption if isinstance(entry, CaptioningBatch) else None for entry in batch],
    )

# Combine two datasets (here even of different type) with weight 60/40%, i.e. batches will consist
# 60% of first dataset, and 40% of second dataset.
mix_dataset = MixBatchDataset(
    # It is important to set the inner batch_size to 1, to ensure proper mixing
    (
        get_train_dataset(
            '/my/dataset/image_dataset',
            batch_size=1,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
            task_encoder=ImageTask(),
        ),
        6,
    ),
    (
        get_train_dataset(
            '/my/dataset/captioning_dataset',
            batch_size=1,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
            task_encoder=CaptioningTask(),
        ),
        4,
    ),
    batch_size=32,
    batch_mix_fn=my_mix_fn,
)

# If needed, epochize here:
#mix_dataset = EpochizeDataset(mix_dataset, num_epochs=10)
# If needed, map the batches
#mix_dataset = MapDataset(mix_dataset, map_fn=lambda batch: batch)

train_loader = get_loader(mix_dataset)

# Iterating over train_loader will yield batches of type ImageWithOptionalCaptionBatch.
for sample in train_loader:
    # sample is ImageWithOptionalCaptionBatch
    pass

```


## Sample weights

You may also introduce weights per sample. I.e. a sample from one dataset may be higher weighted in the loss than from another dataset.

```python
from dataclasses import dataclass
from typing import Callable, List

import torch

from megatron.energon import CaptioningSample, DefaultTaskEncoder


# Type for intermediate batch, after batching operation
@dataclass
class CaptioningRawBatch:
    # (n,)
    __key__: List[str]
    # (n, c, h, w)
    image: torch.Tensor
    # (n,)
    caption: List[str]


# Typing for the resulting batch data
@dataclass
class WeightedCaptioningBatch:
    __keys__: List[str]
    # (n, c, h, w)
    images: torch.Tensor
    # (n, c)
    text: torch.Tensor
    # scalar, as samples between sources are not intermixed in a single gpu batch
    weight: float


# All the typing is optional
class WeightedCaptioningTaskEncoder(
    DefaultTaskEncoder[
        CaptioningSample,
        CaptioningSample,
        CaptioningRawBatch,
        WeightedCaptioningBatch,
    ]
):
    """A simple task encoder for captioning."""

    def __init__(
        self,
        tokenizer: Callable[[List[str]], torch.Tensor],
        max_length: int = 128,
        sample_weight: float = 1,
    ):
        super().__init__(batch_type=CaptioningRawBatch)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sample_weight = sample_weight

    def encode_batch(self, batch_data: CaptioningRawBatch) -> WeightedCaptioningBatch:
        return WeightedCaptioningBatch(
            __keys__=batch_data.__key__,
            images=batch_data.image,
            text=self.tokenizer(batch_data.caption),
            weight=self.sample_weight,
        )

```

Combined with dataset blending:
    
```python
# Interleave batches from two data sources. The batches are sampled 50/50% from each dataset (i.e. the
# batches of each dataset appear equally often), but samples from the first dataset have a higher
# weight.
train_loader = get_loader(BlendDataset(
    (
        get_train_dataset(
            '/my/dataset/captioning_dataset1',
            batch_size=32,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
            task_encoder=WeightedCaptioningTaskEncoder(sample_weight=0.2),
        ),
        0.5,
    ),
    (
        get_train_dataset(
            '/my/dataset/captioning_dataset2',
            batch_size=32,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
            task_encoder=WeightedCaptioningTaskEncoder(sample_weight=0.8),
        ),
        0.5,
    ),
))

for sample in train_loader:
    # sample is WeightedCaptioningBatch
    loss = forward_and_compute_loss(sample.image, sample.text) * sample.weight
    loss.backward()
    optimizer.step()

```
