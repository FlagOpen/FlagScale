<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Remote Dataset

Since version 2.0.0, Megatron Energon supports the use of remote datasets that are stored in an object store bucket with an S3-compatible interface.
This means you can train or validate with your data right from that object store by simply swapping the dataset path for a so-called _rclone URL_.

## Prerequisites

To use this feature, you need to set up an [Rclone](https://rclone.org/) configuration. Rclone is an open source software to manage files on cloud storage. While Energon does not depend on the Rclone software itself, we rely on the same configuration mechanism.

So if you don't like to install or use Rclone, that's fine, but you will need to set up a config file that is compatible. We still recommend using Rclone, since it's a great tool.

Once you set up your config at `~/.config/rclone/rclone.conf`, it may look like this:

```
[coolstore]
type = s3
provider = Other
access_key_id = MY_ACCESS_KEY_ID
secret_access_key = MY_SECRET_ACCESS_KEY
region = us-east-1
endpoint = pdx.s8k.io
```

## The URL syntax

The syntax is a simple as 

```
rclone://RCLONE_NAME/BUCKET/PATH
```

For example:

```
rclone://coolstore/mainbucket/datasets/somedata
```

You can use this URL instead of paths to datasets in

* Functions like `get_train_dataset`, `get_val_dataset`
* Inside [metadataset](metadatasets.md) specifications
* As arguments to `energon prepare` or `energon lint`. Note that those may be slow for remote locations.

Example usage:

```python
ds = get_train_dataset(
    'rclone://coolstore/mainbucket/datasets/somedata',
    batch_size=1,
    shuffle_buffer_size=100,
    max_samples_per_sequence=100,
)
```
