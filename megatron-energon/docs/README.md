<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Building the documentation

To build the documentation, you need sphinx and additional packages:

- sphinx-rtd-theme
- sphinx    
- sphinxcontrib-napoleon
- myst-parser

You can install these like

`pip install sphinx-rtd-theme sphinx sphinxcontrib-napoleon myst-parser sphinx-click`

Use `make html` to build it.

Or use PyCharm by adding a configuration:

    `Run -> Edit Configurations -> Add new Configuration -> Python docs -> Sphinx task`

Use the `src/docs/source` folder as input folder and the `src/docs/build` as output.
