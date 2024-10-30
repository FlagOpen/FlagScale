# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import traceback


def log_exception(_e: Exception, sample, /):
    traceback.print_exc()
    print("-" * 10)

    sample_str = str(sample)
    if len(sample_str) > 400:
        sample_str = sample_str[:200] + "..." + sample_str[-200:]

    print(sample_str)

    print("-" * 10)
