# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import re

split_name_re = re.compile(r"^((?:.*/|)[^.]+)[.]([^/]*)$")
skip_meta_re = re.compile(r"__[^/]*__($|/)")
