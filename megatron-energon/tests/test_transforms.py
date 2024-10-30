# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module tests image transforms."""

import logging
import random
import sys
import unittest
from typing import Any, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from megatron.energon.transforms import *


class TestTransforms(unittest.TestCase):
    # Set up the test fixture
    def setUp(self):
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    def tearDown(self):
        pass

    @staticmethod
    def reset_rng(seed: int = 42):
        """Creates a PIL image with random noise."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    @staticmethod
    def get_test_image(width: int, height: int):
        """Creates a PIL image with random noise."""

        arr = np.zeros((width, height, 3), dtype=np.uint8)

        # Some colorful borders
        arr[0, :, :] = [255, 0, 0]
        arr[:, 0, :] = [255, 255, 0]
        arr[-1, :, :] = [255, 255, 255]
        arr[:, -1, :] = [0, 255, 0]

        # A single white pixel
        if width > 3 and height > 3:
            arr[3, 3, :] = [255, 255, 255]

        # And in the middle some noise
        if width > 10 and height > 10:
            arr[5:-5, 5:-5, :] = np.random.randint(0, 255, (width - 10, height - 10, 3))

        return Image.fromarray(arr)

    @staticmethod
    def get_test_image_soft(width: int, height: int):
        """Creates a PIL image smooth content"""

        arr = np.zeros((width, height, 3), dtype=np.uint8)

        # Fill red channel the image with a smooth gradient from left to right.
        arr[:, :, 0] = np.arange(width)[:, None] / width * 255
        # The same for green from top to bottom:
        arr[:, :, 1] = np.arange(height)[None, :] / height * 255

        return Image.fromarray(arr)

    def _apply_and_compare(
        self, testable_transform, img, atol=2, seed=42, msg=None, only_nonblack=False
    ):
        # Then transform using our method
        merge_transform = MergeTransform([testable_transform])

        self.reset_rng(seed=seed)
        test_result = merge_transform(img)

        # And also transform using torchvision directly
        self.reset_rng(seed=seed)
        ref_result = testable_transform(img)

        # Then compare the sizes and the images contents
        self.assertEqual(test_result.size, ref_result.size)

        # Check that image contents are close
        np_test = np.array(test_result)
        np_ref = np.array(ref_result)

        if only_nonblack:
            nonblack_mask = (np_test > 0) & (np_ref > 0)
            np_test = np_test[nonblack_mask]
            np_ref = np_ref[nonblack_mask]

        # The maximum allowed difference between pixel values is 2 (uint8)
        self.assertTrue(np.allclose(np_test, np_ref, atol=atol), msg=msg)

    def test_resize(self):
        """Tests ResizeMapper"""

        MAX_SIZE = 150
        # These are the different setups we test. Each entry is a tuple of
        # (source size, resize_kwargs)

        size_list = [  # source size (w, h), resize_kwargs
            [(100, 100), {"size": (100, 100)}],
            [(200, 50), {"size": (100, 100)}],
            [(50, 50), {"size": (100, 100)}],
            [(500, 500), {"size": (10, 10)}],
            [(1, 2), {"size": (1, 3)}],  # Scale width by 1.5x
            [(50, 100), {"size": 100, "max_size": MAX_SIZE}],  # Test max_size
        ]

        for source_size, resize_kwargs in size_list:
            logging.info(
                f"Testing Resize with source size {source_size} and resize_kwargs {resize_kwargs}"
            )

            # Create a test image of the given source size
            img = TestTransforms.get_test_image(*source_size)
            transform = T.Resize(**resize_kwargs, interpolation=InterpolationMode.NEAREST)

            self._apply_and_compare(
                transform,
                img,
                msg=f"Resize: source_size={source_size}, resize_kwargs={resize_kwargs}",
            )

    def test_random_resized_crop(self):
        """Tests RandomResizedCropMapper"""

        randcrop = T.RandomResizedCrop(
            90, scale=(0.3, 0.7), ratio=(0.75, 1.3), interpolation=InterpolationMode.BILINEAR
        )
        source_size = (50, 60)

        logging.info(f"Testing RandomResizedCrop with source size {source_size}")

        # Create a test image of the given source size
        img = TestTransforms.get_test_image_soft(*source_size)

        self._apply_and_compare(randcrop, img, msg="RandomResizedCrop")

    def test_random_flip(self):
        source_size = (55, 33)
        img = TestTransforms.get_test_image(*source_size)

        logging.info(f"Testing RandomHorizontalFlip 5 times")
        for idx in range(5):
            randhflip = T.RandomHorizontalFlip(p=0.8)
            self._apply_and_compare(randhflip, img, seed=idx, msg="RandomHorizontalFlip")

        logging.info(f"Testing RandomVerticalFlip 5 times")
        for idx in range(5):
            randvflip = T.RandomVerticalFlip(p=0.8)
            self._apply_and_compare(randvflip, img, seed=idx, msg="RandomVerticalFlip")

    def test_random_rotation(self):
        source_size = (55, 33)
        img = TestTransforms.get_test_image_soft(*source_size)

        logging.info(f"Testing RandomRotation without expand")
        for idx in range(5):
            randrot = T.RandomRotation((-90, 269), interpolation=InterpolationMode.BILINEAR)
            self._apply_and_compare(
                randrot,
                img,
                seed=idx,
                msg="RandomRotation without expand",
            )

        logging.info(f"Testing RandomRotation with expand")
        for idx in range(5):
            randrot = T.RandomRotation(
                (-180, 269), interpolation=InterpolationMode.BILINEAR, expand=True
            )
            self._apply_and_compare(
                randrot,
                img,
                seed=idx,
                msg="RandomRotation with expand",
            )

    def test_random_crop(self):
        source_size = (155, 120)
        img = TestTransforms.get_test_image(*source_size)

        size_list = [  # crop size (w, h)
            (155, 120),  # Same size
            (100, 50),
            3,  # Single int as size
            120,
            (155, 8),  # One dimension same size
        ]

        logging.info(f"Testing RandomCrop")
        for idx, size in enumerate(size_list):
            randcrop = T.RandomCrop(size)
            self._apply_and_compare(
                randcrop,
                img,
                seed=idx,
                msg=f"RandomCrop: crop size={size}",
            )

        # Test `pad_if_needed` (Crop size larger than image size)
        randcrop = T.RandomCrop((500, 500), pad_if_needed=True)
        self._apply_and_compare(randcrop, img)

    def test_random_perspective(self):
        source_size = (128, 133)
        img = TestTransforms.get_test_image_soft(*source_size)

        logging.info(f"Testing RandomPerspective")
        for idx in range(5):
            randpersp = T.RandomPerspective(interpolation=InterpolationMode.BILINEAR)
            self._apply_and_compare(
                randpersp,
                img,
                seed=idx,
                msg=f"RandomPerspective: source_size={source_size}",
                only_nonblack=True,  # Sometimes one pixel is off
            )

    def test_center_crop(self):
        source_size_list = [  # source size (w, h)
            (155, 120),
            (154, 119),
        ]

        crop_size_list = [  # crop size (w, h)
            (155, 120),  # Same size
            (100, 50),
            3,  # Single int as size
            120,
            (200, 50),  # Large than image in x direction
            (50, 200),  # Large than image in y direction
            (200, 200),  # Large than image in both directions
        ]

        logging.info(f"Testing CenterCrop")

        for source_size in source_size_list:
            img = TestTransforms.get_test_image(*source_size)

            for idx, crop_size in enumerate(crop_size_list):
                centcrop = T.CenterCrop(crop_size)
                self._apply_and_compare(
                    centcrop,
                    img,
                    seed=idx,
                    msg=f"CenterCrop: source_size={source_size}, crop_size={crop_size}",
                )

    def test_custom(self):
        """Tests if a custom transform works"""

        source_size = (128, 133)

        class FixedTranslate(CustomTransform):
            """Translates the image by 5 pixels in both x and y direction"""

            def __init__(self):
                pass

            def apply_transform(
                self, matrix: np.ndarray, dst_size: np.ndarray
            ) -> Tuple[Any, Any, Any]:
                matrix = self.translate(5, 5) @ matrix
                return matrix, dst_size, (self.__class__.__name__, (5, 5))

        img = TestTransforms.get_test_image(*source_size)

        merge_transform = MergeTransform([FixedTranslate()])
        test_result = merge_transform(img)

        reference_img = Image.new(img.mode, img.size, (0, 0, 0))
        reference_img.paste(img, (5, 5))

        self.assertTrue(
            np.allclose(np.array(test_result), np.array(reference_img), atol=1),
            msg="FixedTranslate",
        )

    def test_merge(self):
        """Tests if two merged transforms yield the same result.
        Merging RandomCrop and RandomPerspective."""

        source_size = (128, 133)
        img = TestTransforms.get_test_image_soft(*source_size)

        randcrop = T.RandomCrop((70, 70))
        randrot = T.RandomRotation((45, 269), interpolation=InterpolationMode.BILINEAR)

        merge_transform = MergeTransform([randrot, randcrop])
        self.reset_rng(1)
        test_result = merge_transform(img)

        self.reset_rng(1)
        ref_result = randcrop(randrot(img))

        self.assertTrue(
            np.allclose(np.array(test_result), np.array(ref_result), atol=1),
            msg="MergeTransform of RandomRotation and RandomCrop",
        )


if __name__ == "__main__":
    unittest.main()
