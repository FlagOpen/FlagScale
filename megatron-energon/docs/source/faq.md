<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# FAQ
Frequently asked questions, small how-to's, and other useful information.

## What's the best way to combine multiple datasets together?

Use the [Metadataset](metadatasets.md). It allows (recursive) combination of multiple datsets, with a given weight.

## On autoresume, the dataset restarts from the beginning

Use the savable loader and save the dataset state along with the model weights on autoresume. Same when restoring
the state, use the savable loader to restore the dataset state when resuming. This ensures that each sample of an inner
dataset is seen only once per dataset-epoch (when mixing, the inner datasets still ensure an epoch).

## How to customize the blending of datasets through metadatasets

See [Metadataset](metadatasets.md). The Task-Encoder allows changing the blending algorithm.

## Process batches of mixed datasets separately (before mixing samples)

See [Metadataset](metadatasets.md). The Task-Encoder allows changing the blending algorithm and order in which batching and blending happens.
