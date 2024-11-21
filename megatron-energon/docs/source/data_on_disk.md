<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Dataset Format on Disk

The energon library supports loading large multi-modal datasets from disk.
To load the dataset, it must comply with the format described in this section.

A valid energon dataset must contain an `.nv-meta` folder with certain files as shown below.

```
my_dataset
├── .nv-meta
│   ├── dataset.yaml
│   ├── split.yaml
│   └── .info.yaml
├── shards
│   ├── shard_000.tar
│   ├── shard_001.tar
│   ├── ...
```

Note that the `shards` folder is just an example. The shards and their folder can be named differently. 
It does not even need to be a webdataset. [Other formats](#flavors_details) are supported, but the `.nv-meta` structure
is always the same.

## Files in `.nv-meta`
### dataset.yaml 
The `dataset.yaml` contains the dataset definition, i.e. the dataset class to use as loader, optional decoders.
If you want to create such a file, you should consider using the [CLI preparation tool](energon_data_prepare).

Here's an example:
```yaml
__class__: CaptioningWebdataset
__module__: megatron.energon
field_map:
  image: jpg
  caption: txt
```

The `__class__` and `__module__` values help the library construct the correct object.
The `field_map` specifies how the fields from each webdataset sample are mapped to the members of the sample dataclass.

In this example, the dataclass is
```python
@dataclass
class CaptioningSample(Sample):
    image: torch.Tensor
    caption: str
```

In some scenarios, you might need a more advanced way to map samples into the dataclass.
In that case, please check out [this page](advanced_dataformat).

### split.yaml
This file contains the splits (i.e. train, val, test), each a list of the shards for each split.
It can also contain a "denylist" to exclude certain samples or shards from training.
Example:

```yaml
exclude: []
split_parts:
  train:
  - shards/shard_000.tar
  - shards/shard_001.tar
  val:
  - shards/shard_002.tar
  test:
  - shards/shard_003.tar
```

To exclude certain shards or samples, you need to add those to the `exclude` list as follows:

```yaml
exclude:
  - shards/shard_004.tar
  - shards/shard_001.tar/000032
  - shards/shard_001.tar/000032
split_parts:
...
```
The above code excludes the entire shard `004` and two samples from the shard `001`.

### .info.yaml
The hidden info file is auto-generated and contains statistics about each shard.

Example:
```yaml
shard_counts:
  shards/000.tar: 1223
  shards/001.tar: 1420
  shards/002.tar: 1418
  shards/003.tar: 1358
```
