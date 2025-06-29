# setup.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from setuptools import setup, find_packages
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
import torch


class PathManager:
    def __init__(self):
        # torch
        torch_root = os.path.dirname(torch.__file__)
        self.torch_inc = os.path.join(torch_root, "include")
        self.torch_csrc_inc = os.path.join(torch_root, "include/torch/csrc/api/include")
        self.torch_lib = os.path.join(torch_root, "lib")

        # pybind11
        self.pybind_inc = pybind11.get_include()

        # ascend
        ascend_root = "/usr/local/Ascend/latest/aarch64-linux/"
        self.ascend_inc = os.path.join(ascend_root, "include")
        self.ascend_lib = os.path.join(ascend_root, "lib64")

        # my own headers
        self.header = "omni_planner/cpp/include"

        # csrc
        self.sources = [
            "omni_planner/cpp/placement_manager.cpp",
            "omni_planner/cpp/placement_mapping.cpp",
            "omni_planner/cpp/placement_optimizer.cpp",
            "omni_planner/cpp/expert_activation.cpp",
            "omni_planner/cpp/tensor.cpp",
            "omni_planner/cpp/moe_weights.cpp",
            "omni_planner/cpp/utils.cpp",
        ]

        self.check()

    def check(self):
        if not os.path.exists(self.torch_inc):
            raise FileNotFoundError(f"PyTorch include path not found: {self.torch_inc}")
        if not os.path.exists(self.torch_lib):
            raise FileNotFoundError(f"PyTorch lib path not found: {self.torch_lib}")

    def get_include_dirs(self):
        include_dirs = [self.header, self.pybind_inc, self.ascend_inc, self.torch_inc]
        if os.path.exists(self.torch_csrc_inc):
            include_dirs.append(self.torch_csrc_inc)
        return include_dirs

    def get_library_dirs(self):
        return [self.torch_lib, self.ascend_lib]

    def get_extra_link_args(self):
        lib_dirs = self.get_library_dirs()
        link_args = [f"-L{x}" for x in lib_dirs]
        link_args.extend([f"-Wl,-rpath={x}" for x in lib_dirs])
        return link_args


paths = PathManager()


# 定义扩展模块
ext_modules = [
    Pybind11Extension(
        "omni_planner.omni_placement",
        sources=paths.sources,
        include_dirs=paths.get_include_dirs(),
        language='c++',
        extra_compile_args=[
            '-std=c++17',
            '-pthread',
        ],
        extra_link_args=[
            '-pthread',
            '-lascendcl',
            '-ltorch',
            '-ltorch_python',
        ] + paths.get_extra_link_args(),
        library_dirs=paths.get_library_dirs(),
        libraries=['torch', 'torch_python', 'ascendcl']
    ),
]


setup(
    name='omni_placement',  # 包的名称
    version='0.6.15.1a3',  # 包的版本
    description='Package for optimizing MoE layer',  # 简短描述
    packages=find_packages(
        exclude=(
            ".cloudbuild",
            "build",
            "examples",
            "scripts",
            "tests",
            "utils",
            "omni_planner.cpp",
        )
    ),
    install_requires=['numpy',
        'torch',
        'transformers',
        'pybind11',
    ],  # 依赖的其他包
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    include_package_data=True,
)
