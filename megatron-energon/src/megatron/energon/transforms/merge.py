# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Type

import numpy as np
import PIL.Image
import torch
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode, pil_modes_mapping

from megatron.energon.transforms.custom import CustomTransform
from megatron.energon.transforms.mappers import TransformMapper


class MergeTransform(torch.nn.Module):
    transforms: List[TransformMapper]

    def __init__(self, transforms: List[torch.nn.Module]):
        super().__init__()
        self.transforms = [self._transform_to_mapper(t) for t in transforms]
        # Get common interpolation and fill values
        interpolation = None
        interpolation_t = None
        fill = None
        fill_t = None
        for t in self.transforms:
            t_fill = t.fill()
            if t_fill is not None:
                if fill is None:
                    fill = t_fill
                    fill_t = t
                if fill is not None and t_fill != fill:
                    raise ValueError(
                        f"Fill values are not equal: {fill} ({fill_t}) != {t_fill} ({t})"
                    )
            t_interpolation = t.interpolation()
            if t_interpolation is not None:
                if interpolation is None:
                    interpolation = t_interpolation
                    interpolation_t = t
                if interpolation is not None and t_interpolation != interpolation:
                    raise ValueError(
                        f"Interpolation values are not equal: {interpolation} ({interpolation_t}) != {t_interpolation} ({t})"
                    )

        self.interpolation = InterpolationMode.BILINEAR if interpolation is None else interpolation
        self.fill_value = fill

    def _transform_to_mapper(self, transform: torch.nn.Module) -> Type[TransformMapper]:
        """Given a transform object, instantiate the corresponding mapper.
        This also handles objects of derived transform classes."""

        if isinstance(transform, CustomTransform):
            # Custom transforms can be used as-is, they provide the apply_transform method
            return transform

        for m in TransformMapper.__subclasses__():
            if isinstance(transform, m.source_type):
                return m(transform)  # Instantiate
        raise ValueError(f"Unsupported transform type {type(transform)}")

    def forward(self, x):
        matrix = np.eye(3, dtype=np.float64)
        if isinstance(x, PIL.Image.Image):
            dst_size = np.array((x.height, x.width), dtype=np.int64)
        else:
            dst_size = np.array(x.shape[-2:], dtype=np.int64)
        all_params = []
        for transform in self.transforms:
            matrix, dst_size, params = transform.apply_transform(matrix, dst_size)
            all_params.append(params)

        if isinstance(x, PIL.Image.Image):
            try:
                interpolation = pil_modes_mapping[self.interpolation]
            except KeyError:
                raise NotImplementedError(f"interpolation: {self.interpolation}")

            # Invert matrix for backward mapping
            matrix = np.linalg.inv(matrix)

            # Scale matrix
            matrix /= matrix[2, 2]

            if self.fill_value is None:
                fill_color = None
            elif isinstance(self.fill_value, (int, float)):
                fill_color = (self.fill_value,) * len(x.getbands())
            else:
                fill_color = self.fill_value
            if np.allclose(matrix[2, :2], [0, 0]):
                # print("PIL Affine")
                return x.transform(
                    tuple(dst_size[::-1]),
                    PIL.Image.AFFINE,
                    matrix.flatten()[:6],
                    interpolation,
                    fillcolor=fill_color,
                )
            else:
                # print("PIL Perspective")
                return x.transform(
                    tuple(dst_size[::-1]),
                    PIL.Image.PERSPECTIVE,
                    matrix.flatten()[:8],
                    interpolation,
                    fillcolor=fill_color,
                )
        elif isinstance(x, torch.Tensor):
            print("torch affine")
            if self.interpolation == T.InterpolationMode.NEAREST:
                interpolation = "nearest"
            elif self.interpolation == T.InterpolationMode.BILINEAR:
                interpolation = "bilinear"
            elif self.interpolation == T.InterpolationMode.BICUBIC:
                interpolation = "bicubic"
            else:
                raise NotImplementedError(f"interpolation: {self.interpolation}")
            if self.fill_value is not None and self.fill_value != 0:
                raise NotImplementedError(
                    f"Fill value {self.fill_value} is not supported for torch"
                )
            # Normalize to [-1, 1] range
            matrix = (
                TransformMapper.translate(-1, -1)
                @ TransformMapper.scale(2 / dst_size[1], 2 / dst_size[0])
                @ matrix
                @ TransformMapper.scale(x.shape[-1] / 2, x.shape[-2] / 2)
                @ TransformMapper.translate(1, 1)
            )

            matrix = np.linalg.inv(matrix)
            if np.allclose(matrix[2, :2], [0, 0]):
                grid = torch.nn.functional.affine_grid(
                    torch.as_tensor(matrix[None, :2, :], dtype=torch.float32),
                    torch.Size((1, 3, *dst_size)),
                )
            else:
                xs = torch.linspace(-1, 1, dst_size[1], dtype=torch.float32)
                ys = torch.linspace(-1, 1, dst_size[0], dtype=torch.float32)
                zs = torch.ones((1,), dtype=torch.float32)
                # shape: (2<x,y,1>, W, H)
                grid = torch.stack(torch.meshgrid([xs, ys, zs], indexing="ij"))[..., 0]
                # shape: (H, W, 2<x,y,1>)
                grid = grid.permute(2, 1, 0)
                # shape: (H, W, 3<x,y,w>, 1)
                grid = (
                    torch.as_tensor(matrix, dtype=torch.float32)[None, None, ...] @ grid[..., None]
                )
                # shape: (H, W, 2<x,y>)
                grid = grid[:, :, :2, 0] / grid[:, :, 2:3, 0]
                # shape: (1, H, W, 2<x,y>)
                grid = grid[None, ...]
            return torch.nn.functional.grid_sample(
                x[None, ...], grid, interpolation, padding_mode="zeros", align_corners=False
            )[0, ...]
        else:
            raise NotImplementedError()
            # TODO: Needs implementation and testing
            import cv2

            return cv2.warpAffine(x, matrix[:2], tuple(dst_size), flags=cv2.INTER_LINEAR)
