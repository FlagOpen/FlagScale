<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Installation

If you simply want to use this package without modifying it, the best option is to install it 
as a dependency of your project like you would with any other pip package.

## Normal Installation

To install the most recent version, run

```shell
pip install megatron-energon
```

in your project's Python environment, which could be a virtualenv, or a conda environment.
You can even install it inside a `Dockerfile` to include it in your custom docker container.

## Installation from Source Code

If you want to manage, debug or modify the code of the energon package, we recommend that you clone this repository
on your disk or even as a submodule of your project.
You can then install the package in "development" mode in-place. This way, the package will not be hidden inside
your pip package management, but will stay in the location where you cloned it and you can even modify it in-place.

To check out locally and install in development mode:
```shell
git clone https://github.com/NVIDIA/Megatron-Energon.git megatron-energon
pip install -e ./megatron-energon
```

Or to add as a submodule to your project and install in development mode:
```shell
git submodule add https://github.com/NVIDIA/Megatron-Energon.git megatron-energon
pip install -e ./megatron-energon
```

```{warning}
**We discourage importing the cloned repo without pip install** 
- You will not be able to use the command line tool
- You would have to use hacks to get the package into your `PYTHONPATH`
- You would need to take care of the dependencies yourself. 

Instead, simply install in development mode.
```
