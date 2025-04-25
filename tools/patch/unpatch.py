import argparse
import os
import shutil
import sys
import tempfile

from git.repo import Repo

DELETED_FILE_NAME = "deleted_files.txt"


def unpatch(src, dst, submodule_name, mode="symlink"):
    """Unpatch the backend with symlinks."""
    print(f"Unpatching backend {submodule_name}...")
    init_submodule(dst, submodule_name)
    assert mode in ["symlink", "copy"]
    if mode == "copy":
        _copy(src, dst)
    elif mode == "symlink":
        _create_symlinks(src, dst)
    deleted_files_path = os.path.join(src, DELETED_FILE_NAME)
    if os.path.lexists(deleted_files_path):
        _delete_file(deleted_files_path, dst)


def _copy(src, dst):
    for root, dirs, files in os.walk(src):
        for filename in files:
            src_file = os.path.join(root, filename)
            if src_file == os.path.join(src, DELETED_FILE_NAME):
                continue

            rel_path = os.path.relpath(src_file, src)
            dst_file = os.path.join(dst, rel_path)

            dst_file_dir = os.path.dirname(dst_file)
            if not os.path.lexists(dst_file_dir):
                os.makedirs(dst_file_dir)

            if os.path.lexists(dst_file):
                os.remove(dst_file)
            assert not os.path.lexists(dst_file)
            shutil.copyfile(src_file, dst_file)
            print(f"Copied file: {dst_file} -> {src_file}")


def _delete_file(file_path, dst):
    with open(file_path, "r", encoding="utf-8") as f:
        deleted_files = f.readlines()
        for deleted_file in deleted_files:
            deleted_file = deleted_file.strip()
            deleted_file_path = os.path.join(dst, deleted_file)
            if os.path.lexists(deleted_file_path):
                os.remove(deleted_file_path)
                print(f"Deleted file: {deleted_file_path}")
            else:
                print(f"File not found for deletion: {deleted_file_path}")


def _create_symlinks(src, dst):
    for root, dirs, files in os.walk(src):
        for filename in files:
            src_file = os.path.join(root, filename)
            if src_file == os.path.join(src, DELETED_FILE_NAME):
                continue

            rel_path = os.path.relpath(src_file, src)
            dst_file = os.path.join(dst, rel_path)

            dst_file_dir = os.path.dirname(dst_file)
            if not os.path.lexists(dst_file_dir):
                os.makedirs(dst_file_dir)

            if os.path.lexists(dst_file):
                os.remove(dst_file)
            assert not os.path.lexists(dst_file)
            os.symlink(src_file, dst_file)

            print(f"Created symbolic link: {dst_file} -> {src_file}")


def init_submodule(dst, submodule_name):
    if os.path.lexists(dst) and len(os.listdir(dst)) > 0:
        print(f"Skipping {submodule_name} initialization, as it already lexists.")
        return
    print(f"Initializing submodule {submodule_name}...")
    repo = Repo(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    submodule = repo.submodule(submodule_name)
    submodule.update(init=True)
    print(f"Initialized {submodule_name} submodule.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch or unpatch backend with symlinks.")
    parser.add_argument(
        "--backend",
        nargs="+",
        choices=["Megatron-LM", "vllm"],
        default=["Megatron-LM"],
        help="Backend to unpatch (default: Megatron-LM)",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Mode to unpatch (default: symlink)",
    )

    args = parser.parse_args()
    backends = args.backend
    if not isinstance(backends, list):
        backends = [backends]
    # FlagScale/tools/patch
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # FlagScale/tools
    script_dir = os.path.dirname(script_dir)
    # FlagScale
    main_path = os.path.dirname(script_dir)

    for backend in backends:
        submodule_name = f"third_party/{backend}"
        src = None
        dst = os.path.join(main_path, "third_party", backend)
        # Megatron-LM
        if backend == "Megatron-LM":
            src = os.path.join(main_path, "flagscale", "train", "backends", backend)
            assert src
            unpatch(src, dst, submodule_name, mode=args.mode)

        # vllm
        if backend == "vllm":
            src = os.path.join(main_path, "flagscale", "inference", "backends", backend)
            assert src
            unpatch(src, dst, submodule_name, mode=args.mode)
