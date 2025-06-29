# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) Huawei Technologies Co., Ltd. 2025.All rights reserved.

""" define marks """
import pytest
import numpy as np


def arg_mark(plat_marks, level_mark):
    optional_plat_marks = ['platform_ascend', 'platform_ascend910b', 'platform_ascend310p', 'platform_gpu',
                           'cpu_linux', 'cpu_windows', 'cpu_macos']
    # level0：门禁级别/冒烟用例，level1:全量用例，lts：长稳，common：通用
    optional_level_marks = ['level0', 'level1', 'lts', 'common']
    if not plat_marks or not set(plat_marks).issubset(set(optional_plat_marks)):
        raise ValueError("wrong plat_marks values")
    if level_mark not in optional_level_marks:
        raise ValueError("wrong level_mark value")

    def decorator(func):
        for plat_mark in plat_marks:
            func = getattr(pytest.mark, plat_mark)(func)
        func = getattr(pytest.mark, level_mark)(func)
        set_numpy_global_seed()
        return func

    return decorator


def get_numpy_global_seed():
    return 1967515154


def set_numpy_global_seed():
    np.random.seed(get_numpy_global_seed())



