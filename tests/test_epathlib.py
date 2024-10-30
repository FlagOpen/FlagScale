# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module tests the EPath class, our custom version of pathlib.Path"""

import logging
import struct
import sys
import unittest

from megatron.energon.epathlib import EPath
from megatron.energon.epathlib.rclone_config import ConfigEntry


class TestEPath(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    def tearDown(self):
        pass

    def test_basic(self):
        """Some basic functionality tests"""

        p_rel = EPath("./subdir")
        p_abs = EPath("/tmp")

        p_comb = p_abs / p_rel
        # logging.info(f"p_comb: {p_comb}")
        # logging.info(f"p_comb: {p_comb.internal_path}")

        # We don't want to work on relative paths
        self.assertRaises(AssertionError, lambda: p_rel.is_file())

        # Those should not raise:
        assert p_comb.is_absolute()
        _ = p_comb.is_file()
        _ = p_abs.is_file()

    def test_contextman(self):
        """Test the context manager"""

        tmp_file_path = "/tmp/testfile.bin"
        # First create a file
        with open(tmp_file_path, "wb") as f:
            f.write(struct.pack("H10s", 1337, b"1234567890"))

        # Test context manager reading
        p = EPath(tmp_file_path).open("rb")
        with p:
            b = p.read()
            assert isinstance(b, bytes)

            num, data = struct.unpack("H10s", b)
            logging.info(f"num: {num}")
            assert num == 1337
            assert data == b"1234567890"

            assert not p.closed

        assert p.closed

        # Test context manager writing
        tmp_file_path2 = "/tmp/testfile2.bin"
        with EPath(tmp_file_path2).open("wb") as p:
            p.write(struct.pack("H10s", 1337, b"1234567890"))

    def test_glob(self):
        """Test the glob functionality"""

        # First create some files
        for i in range(10):
            with open(f"/tmp/epathtestfile_{i}.bin", "wb") as f:
                f.write(b"dummycontent")

        # Test globbing
        p = EPath("/tmp").glob("epathtestfile_*.bin")

        logging.info(f"p: {p}, type of p: {type(p)}")
        elems = list(p)
        assert len(elems) == 10
        for i, e in enumerate(elems):
            logging.info(f"glob_result[{i}]: {e}")
            assert isinstance(e, EPath)
            assert e.is_file()

        # Test globbing with a pattern
        p = EPath("/tmp").glob("epathtestfile_[0-3].bin")
        assert len(list(p)) == 4

    def test_s3_path_resolution(self):
        """Test s3 path resolution"""
        config_override = {
            "s3": ConfigEntry(
                name="s3",
                type="s3",
                provider="s3",
                access_key_id="dummy",
                secret_access_key="dummy",
                region="dummy",
                endpoint="https://localhost",
            )
        }

        # Test globbing
        p = EPath("rclone://s3/tmp/path/subpath.txt", config_override=config_override)
        assert str(p) == "rclone://s3/tmp/path/subpath.txt", str(p)

        p2 = p / ".." / "subpath2.txt"
        assert str(p2) == "rclone://s3/tmp/path/subpath2.txt", str(p2)

        p3 = EPath("rclone://s3/tmp/path/.././subpath.txt", config_override=config_override)
        assert str(p3) == "rclone://s3/tmp/subpath.txt", str(p3)

        p4 = p3.parent / "../bla/bla/bla/../../../no/../subpath2.txt"
        assert str(p4) == "rclone://s3/subpath2.txt", str(p4)


if __name__ == "__main__":
    unittest.main()
