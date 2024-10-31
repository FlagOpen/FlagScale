# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import math
import numbers
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torchvision import transforms as T

from megatron.energon.transforms.common import TTransform


class TransformMapper(ABC, Generic[TTransform]):
    source_type: ClassVar[Type[torch.nn.Module]]

    transform: TTransform

    def __init__(self, transform: TTransform):
        self.transform = transform

    def __call__(self, sample):
        return self.transform(sample)

    @staticmethod
    def translate(x: float, y: float) -> np.ndarray:
        m = np.eye(3, dtype=np.float64)
        m[0, 2] = x
        m[1, 2] = y
        return m

    @staticmethod
    def rotate(angle: float) -> np.ndarray:
        """Counter-clockwise rotation. Note that the Y-axis is point down."""
        m = np.eye(3, dtype=np.float64)
        m[:2, :2] = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        return m

    @staticmethod
    def scale(x: float, y: float) -> np.ndarray:
        m = np.eye(3, dtype=np.float64)
        m[0, 0] = x
        m[1, 1] = y
        return m

    @staticmethod
    def shear(x: float, y: float) -> np.ndarray:
        m = np.eye(3, dtype=np.float64)
        m[0, 1] = x
        m[1, 0] = y
        return m

    @staticmethod
    def hflip() -> np.ndarray:
        m = np.eye(3, dtype=np.float64)
        m[0, 0] = -1
        return m

    @staticmethod
    def vflip() -> np.ndarray:
        m = np.eye(3, dtype=np.float64)
        m[1, 1] = -1
        return m

    @abstractmethod
    def apply_transform(
        self, matrix: np.ndarray, dst_size: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Any]: ...

    def fill(
        self,
    ) -> Optional[Union[int, float, Tuple[Union[int, float], ...], List[Union[int, float]]]]:
        return None

    def interpolation(self) -> Optional[T.InterpolationMode]:
        return None


class ResizeMapper(TransformMapper[T.Resize]):
    source_type = T.Resize

    def __init__(self, transform: T.Resize):
        super().__init__(transform)

    def _compute_resized_output_size(
        self, image_size: Tuple[int, int], size: List[int], max_size: Optional[int] = None
    ) -> List[int]:
        if len(size) == 1:  # specified size only for the smallest edge
            h, w = image_size
            short, long = (w, h) if w <= h else (h, w)
            requested_new_short = size[0]

            new_short, new_long = requested_new_short, int(requested_new_short * long / short)

            if max_size is not None:
                if max_size <= requested_new_short:
                    raise ValueError(
                        f"max_size = {max_size} must be strictly greater than the requested "
                        f"size for the smaller edge size = {size}"
                    )
                if new_long > max_size:
                    new_short, new_long = int(max_size * new_short / new_long), max_size

            new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
        else:  # specified both h and w
            new_w, new_h = size[1], size[0]
        return [new_h, new_w]

    def apply_transform(
        self, matrix: np.ndarray, dst_size: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[Any, ...]]:
        size = self.transform.size

        if isinstance(size, int):
            size = [size]

        h, w = self._compute_resized_output_size(dst_size, size, self.transform.max_size)

        matrix = self.scale(w / dst_size[1], h / dst_size[0]) @ matrix
        # matrix = self.scale((w - 1) / (dst_size[1] - 1), (h - 1) / (dst_size[0] - 1)) @ matrix
        # matrix = self.translate(0.25, 0.25) @ matrix
        # matrix = self.translate(0.1, 0) @ matrix
        dst_size = np.array((h, w), dtype=dst_size.dtype)
        # print(f"Resize s={size}")
        return matrix, dst_size, (self.source_type.__name__, size)

    def interpolation(self) -> Optional[T.InterpolationMode]:
        return self.transform.interpolation


class RandomResizedCropMapper(TransformMapper[T.RandomResizedCrop]):
    source_type = T.RandomResizedCrop

    def __init__(self, transform: T.RandomResizedCrop):
        super().__init__(transform)

    def get_params(self, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Gets the parameters for a random resized crop.
        This function is derived from T.RandomResizedCrop.get_params, but without requiring the
        input image (to determine the input size).

        Returns:
            Tuple of (top, left, height, width).
        """
        height, width = size
        area = height * width

        log_ratio = torch.log(torch.tensor(self.transform.ratio))
        for _ in range(10):
            target_area = (
                area
                * torch.empty(1).uniform_(self.transform.scale[0], self.transform.scale[1]).item()
            )
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.transform.ratio):
            w = width
            h = int(round(w / min(self.transform.ratio)))
        elif in_ratio > max(self.transform.ratio):
            h = height
            w = int(round(h * max(self.transform.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def apply_transform(
        self, matrix: np.ndarray, dst_size: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[Any, ...]]:
        top, left, height, width = self.get_params(dst_size)
        # print(
        #     "RandomResizedCrop", top, left, dst_size[0] - height - top, dst_size[1] - width - left
        # )
        # Crop to left, top, height, width
        matrix = self.translate(-left, -top) @ matrix
        dst_size = np.array([height, width], dtype=dst_size.dtype)
        # Resize to target size
        matrix = (
            self.scale(self.transform.size[1] / dst_size[1], self.transform.size[0] / dst_size[0])
            @ matrix
        )
        dst_size = np.array(self.transform.size, dtype=dst_size.dtype)
        return matrix, dst_size, (self.source_type.__name__, (top, left, height, width))

    def interpolation(self) -> Optional[T.InterpolationMode]:
        return self.transform.interpolation


class RandomHorizontalFlipMapper(TransformMapper[T.RandomHorizontalFlip]):
    source_type = T.RandomHorizontalFlip

    def __init__(self, transform: T.RandomHorizontalFlip):
        super().__init__(transform)

    def apply_transform(
        self, matrix: np.ndarray, dst_size: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        do_flip = torch.rand(1) < self.transform.p
        if do_flip:
            matrix = self.hflip() @ matrix
            matrix = self.translate(dst_size[1], 0) @ matrix
            # print(f"RandomHorizontalFlip")
        return matrix, dst_size, (self.source_type.__name__, do_flip)


class RandomVerticalFlipMapper(TransformMapper[T.RandomVerticalFlip]):
    source_type = T.RandomVerticalFlip

    def __init__(self, transform: T.RandomVerticalFlip):
        super().__init__(transform)

    def apply_transform(
        self, matrix: np.ndarray, dst_size: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        do_flip = torch.rand(1) < self.transform.p
        if do_flip:
            matrix = self.vflip() @ matrix
            matrix = self.translate(0, dst_size[0]) @ matrix
            # print(f"RandomVerticalFlip")
        return matrix, dst_size, (self.source_type.__name__, do_flip)


class RandomRotationMapper(TransformMapper[T.RandomRotation]):
    source_type = T.RandomRotation

    def __init__(self, transform: T.RandomRotation):
        super().__init__(transform)

    def apply_transform(
        self, matrix: np.ndarray, dst_size: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        assert self.transform.center is None, "Only centered rotation is supported"
        degrees = self.transform.get_params(self.transform.degrees)
        rads = degrees * np.pi / 180
        # print(f"Rotate deg={degrees}")
        orig_size = dst_size
        if self.transform.expand:
            # Compute size of rotated rectangle
            w = np.abs(np.sin(rads)) * dst_size[0] + np.abs(np.cos(rads)) * dst_size[1]
            h = np.abs(np.sin(rads)) * dst_size[1] + np.abs(np.cos(rads)) * dst_size[0]

            # Round in the same way as PIL does
            rounded_w = np.ceil(orig_size[1] / 2 + w / 2) - np.floor(orig_size[1] / 2 - w / 2)
            rounded_h = np.ceil(orig_size[0] / 2 + h / 2) - np.floor(orig_size[0] / 2 - h / 2)

            # New size is h, w
            dst_size = np.array([int(rounded_h), int(rounded_w)], dtype=dst_size.dtype)
        matrix = (
            self.translate(dst_size[1] / 2, dst_size[0] / 2)
            @ self.rotate(rads)
            @ self.translate(-orig_size[1] / 2, -orig_size[0] / 2)
            @ matrix
        )
        return matrix, dst_size, (self.source_type.__name__, degrees)

    def fill(
        self,
    ) -> Optional[Union[int, float, Tuple[Union[int, float], ...], List[Union[int, float]]]]:
        return self.transform.fill

    def interpolation(self) -> Optional[T.InterpolationMode]:
        return self.transform.interpolation


class RandomCropMapper(TransformMapper[T.RandomCrop]):
    source_type = T.RandomCrop

    def __init__(self, transform: T.RandomCrop):
        super().__init__(transform)

    def apply_transform(
        self, matrix: np.ndarray, dst_size: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        th, tw = self.transform.size  # Target height and width

        # pad the width if needed
        if self.transform.pad_if_needed and dst_size[1] < tw:
            padding = tw - dst_size[1]  # Pad this much on both left and right
            matrix = self.translate(padding, 0) @ matrix
            dst_size[1] += 2 * padding

        # pad the height if needed
        if self.transform.pad_if_needed and dst_size[0] < th:
            padding = th - dst_size[0]  # Pad this much on both top and bottom
            matrix = self.translate(0, padding) @ matrix
            dst_size[0] += 2 * padding

        h, w = dst_size

        if h < th or w < tw:
            raise ValueError(
                f"Required crop size {(th, tw)} is larger than input image size {(h, w)}"
            )

        if w == tw and h == th:
            # No need to crop if we're at the target size already
            i = 0
            j = 0
        else:
            i = torch.randint(0, h - th + 1, size=(1,)).item()  # Offset y
            j = torch.randint(0, w - tw + 1, size=(1,)).item()  # Offset x
            matrix = self.translate(-j, -i) @ matrix

        if self.transform.pad_if_needed:
            dst_size = np.array((th, tw), dtype=dst_size.dtype)
        else:
            dst_size = np.array((min(th, dst_size[0]), min(tw, dst_size[1])), dtype=dst_size.dtype)
        # print(f"RandomCrop t=[{dx}, {dy}], s={dst_size}")
        return matrix, dst_size, (self.source_type.__name__, (j, i, th, tw))

    def fill(
        self,
    ) -> Optional[Union[int, float, Tuple[Union[int, float], ...], List[Union[int, float]]]]:
        return self.transform.fill


class RandomPerspectiveMapper(TransformMapper[T.RandomPerspective]):
    source_type = T.RandomPerspective

    def __init__(self, transform: T.RandomPerspective):
        super().__init__(transform)

    @staticmethod
    def compute_homography(
        startpoints: List[Tuple[float, float]], endpoints: List[Tuple[float, float]]
    ) -> np.ndarray:
        assert len(startpoints) == len(endpoints) == 4

        a_matrix = torch.zeros(2 * len(startpoints), 8, dtype=torch.float)

        for i, (p1, p2) in enumerate(zip(endpoints, startpoints)):
            a_matrix[2 * i, :] = torch.tensor(
                [p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]]
            )
            a_matrix[2 * i + 1, :] = torch.tensor(
                [0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]]
            )

        b_matrix = torch.tensor(startpoints, dtype=torch.float).view(8)
        res = torch.linalg.lstsq(a_matrix, b_matrix, driver="gels").solution

        m = np.eye(3, dtype=np.float32)
        m[0, :] = res[:3]
        m[1, :] = res[3:6]
        m[2, :2] = res[6:]

        return m

    def apply_transform(
        self, matrix: np.ndarray, dst_size: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        assert self.transform.fill == 0, "Only zero fill is supported"
        startpoints = None
        endpoints = None
        if torch.rand(1) <= self.transform.p:
            startpoints, endpoints = self.transform.get_params(
                dst_size[1], dst_size[0], self.transform.distortion_scale
            )
            # print(
            #     f"Perspective ds={self.transform.distortion_scale}: sp={startpoints} -> ep={endpoints}"
            # )
            matrix = self.compute_homography(endpoints, startpoints) @ matrix
        return matrix, dst_size, (self.source_type.__name__, startpoints, endpoints)

    def fill(
        self,
    ) -> Optional[Union[int, float, Tuple[Union[int, float], ...], List[Union[int, float]]]]:
        return self.transform.fill

    def interpolation(self) -> Optional[T.InterpolationMode]:
        return self.transform.interpolation


class CenterCropMapper(TransformMapper[T.CenterCrop]):
    source_type = T.CenterCrop

    def __init__(self, transform: T.CenterCrop):
        super().__init__(transform)

    def apply_transform(
        self, matrix: np.ndarray, dst_size: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        if isinstance(self.transform.size, numbers.Number):
            th, tw = int(self.transform.size), int(self.transform.size)
        elif isinstance(self.transform.size, (tuple, list)) and len(self.transform.size) == 1:
            th, tw = self.transform.size[0], self.transform.size[0]
        else:
            th, tw = self.transform.size

        shift_y = round(float(th - dst_size[0]) / 2)
        shift_x = round(float(tw - dst_size[1]) / 2)

        matrix = self.translate(shift_x, shift_y) @ matrix
        dst_size = np.array((th, tw), dtype=dst_size.dtype)
        # print(f"CenterCrop t=[{dx}, {dy}], s={dst_size}")
        return matrix, dst_size, (self.source_type.__name__, (shift_y, shift_x, th, tw))
