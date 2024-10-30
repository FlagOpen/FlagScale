<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Reproducible Scaling

A special use case is to re-run or continue a training run with the exact same data order, but using a different number of nodes or ranks.

Since version 2.0.0, Megatron Energon supports this behavior if a few constraints are met:

* The global batch size must stay the same across runs
* The global batch size must be a multiple of `micro-batch size * world_size * num_workers`
  * The multiple of that is the number of gradient accumulation steps in your training
* The product `world_size * num_workers` must stay the same across runs, such that the global number of workers stays the same
* You need to set the same `torch.manual_seed(...)` on each rank before constructing the dataset and the data loader

By obeying these rules, you will be able to reproduce the same global batches. Let's look at an example.

| Name  | Global batch size | Micro batch size | World size | Number of Workers | Gradient accumulation steps |
| ----- | ----------------- | ---------------- | ---------- | ----------------- | --------------------------- |
| Run 1 | 8                 | 2                | 4          | 1                 | 1                           |
| Run 2 | 8                 | 2                | 1          | 4                 | 4                           |

Iterating the dataset will yield the same global batches for both of these runs, if the seed is set correctly.

In practice, you will need to adapt your worker config accordingly.

