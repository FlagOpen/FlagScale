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


def validate_device_type(device_type, main_path):
    if device_type:
        if device_type.count("_") != 1 or len(device_type.split("_")) != 2:
            raise ValueError(
                "Invalid format. Device type must be in the format xxx_yyy."
            )
        sorted_backends = sorted(backends)
        backend_key = "+".join(sorted_backends)
        device_path = os.path.join(main_path, "hardware", device_type)
        if not os.path.exists(device_path):
            raise ValueError(f"Please check the hardware directory {device_path}.")
        path_exist = False
        patch_dir = None
        for task_name in os.listdir(device_path):
            task_path = os.path.join(device_path, task_name)
            if os.path.isdir(task_path):
                path = os.path.join(task_path, backend_key)
                if os.path.isdir(path):
                    path_exist = True
                    patch_dir = path
                    break
        if not path_exist:
            raise ValueError(
                f"The patch file for this backend {backend_key} of this hardware {device_type} was not found.."
            )
        else:
            error = f"The files in this directory {patch_dir} must be a file with a .patch suffix and a file with a .yaml suffix."
            if len(os.listdir(patch_dir)) != 2:
                raise ValueError(error)
            base_commit_id = None
            for file in os.listdir(patch_dir):
                if not file.endswith(".patch") and not file.endswith(".yaml"):
                    raise ValueError(error)
                base_commit_id = file.split(".")[0]
            main_repo = Repo(main_path)
            try:
                main_repo.commit(base_commit_id)
            except ValueError:
                raise ValueError(
                    f"The commit ID {base_commit_id} does not exist in the FlagScale."
                )
        return patch_dir, base_commit_id
    return None


def apply_hardware_patch_file(patch_dir, main_path):
    patch_dir, base_commit_id = validate_result
    patch_file = os.path.join(patch_dir, f"{base_commit_id}.patch")
    build_path = os.path.join(main_path, "build", device_type)
    final_path = os.path.join(build_path, os.path.basename(main_path))

    try:
        # Step 1: Remove existing build directory if present
        if os.path.exists(build_path):
            print(f"Removing existing build path: {build_path}")
            shutil.rmtree(build_path)

        # Step 2: Copy main_path to a temporary directory
        temp_path = tempfile.mkdtemp()
        print(f"Copying {main_path} to temp path {temp_path}")
        shutil.copytree(main_path, temp_path, dirs_exist_ok=True)

        # Step 3: Checkout the base commit
        repo = Repo(temp_path)
        print(f"Checking out {base_commit_id} in temp path {temp_path}")

        repo.git.checkout(base_commit_id)

        # Step 4: Apply the patch
        print(f"Applying patch: {patch_file}")
        repo.git.apply("--index", "--whitespace", "fix", patch_file)

        # Step 5: Move the patched temp directory to build/<device_type>/
        print(f"Moving patched temp path to {final_path}")
        os.makedirs(build_path, exist_ok=True)
        shutil.move(temp_path, final_path)

        # Step 6: Update main_path
        main_path = final_path

    except Exception as e:
        print(f"Exception occurred: {e}")

        # Clean up temp directory
        if "temp_path" in locals() and os.path.exists(temp_path):
            print(f"Cleaning up temp path: {temp_path}")
            shutil.rmtree(temp_path, ignore_errors=True)

        # Clean up build directory
        if os.path.exists(build_path):
            print(f"Cleaning up build path: {build_path}")
            shutil.rmtree(build_path, ignore_errors=True)

        raise e
    return final_path


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
    parser.add_argument(
        "--device-type",
        type=str,
        default=None,
        help="Device type. Default is None.",
    )

    args = parser.parse_args()
    backends = args.backend
    device_type = args.device_type

    if not isinstance(backends, list):
        backends = [backends]
    # FlagScale/tools/patch
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # FlagScale/tools
    script_dir = os.path.dirname(script_dir)
    # FlagScale
    main_path = os.path.dirname(script_dir)

    # Check patch exist
    validate_result = validate_device_type(device_type, main_path)
    if validate_result is not None:
        main_path = apply_hardware_patch_file(validate_result, main_path)

    for backend in backends:
        submodule_name = f"third_party/{backend}"
        dst = os.path.join(main_path, "third_party", backend)
        src = os.path.join(main_path, "flagscale", "backends", backend)
        unpatch(src, dst, submodule_name, mode=args.mode)
