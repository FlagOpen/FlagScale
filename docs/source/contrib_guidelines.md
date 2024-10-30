<!--- Copyright (c) 2024, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Contribution Guidelines

If you want to contribute to this repository please adhere to the following guidelines

- Always use [black](https://pypi.org/project/black/) to format your code before committing
- Python `@dataclass` and `NamedTuple` are preferred over dictionaries, which don't allow for IDE
  auto-completion and type checking
- User-exposed classes and methods should be documented in Google-style docstrings that are parsed by sphinx
  and end up in this documentation
