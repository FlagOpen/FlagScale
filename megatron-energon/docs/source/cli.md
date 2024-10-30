<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Command-Line Interface

Once the package is installed, a script called `energon` will be added to your PATH.
Currently, it provides mainly commands to prepare, convert, or lint datasets on disk.

Later, it may be extended for model or checkpoint management for example.

Here's a simple example:

```shell
energon prepare /mnt/data/my_captioning_webdataset
```

The above command will scan your existing off-the-shelf [web dataset](https://webdataset.github.io/webdataset/)
and add the [needed metadata](data_on_disk) to make it compatible with Energon. 

Below, you can see the available sub-commands under `energon`.


```{eval-rst}
.. click:: megatron.energon.cli.main:data_command
   :prog: energon
   :nested: short
```

(energon_data_prepare)=
## energon prepare

An interactive tool to generate metadata for your existing webdataset.
This will help make the dataset compliant with our [format](data_on_disk).

The tool will ask you for a train/val/test split and how to assign the webdataset fields to the
fields of the corresponding sample type in Energon.

See [Data Preparation](data_prep) for more details on how to use this command.


## energon lint

You can execute this tool on the prepared dataset to check if the data is valid and loadable.
It will report any problems such as non-readable images.
