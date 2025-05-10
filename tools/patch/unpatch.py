import argparse
import logging
import os
import shutil
import sys
import tempfile

import yaml

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from git.repo import Repo

DELETED_FILE_NAME = "deleted_files.txt"
FLAGSCALE_BACKEND = "FlagScale"


logger = logging.getLogger("FlagScaleUnpatchLogger")
logger.setLevel(logging.INFO)


if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[FlagScale-Unpatch] %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


def unpatch(main_path, src, dst, submodule_name, mode="symlink", force=False):
    """Unpatch the backend with symlinks."""
    if submodule_name.split("/")[-1] != FLAGSCALE_BACKEND:
        logger.info(f"Unpatching backend {submodule_name}...")
        init_submodule(main_path, dst, submodule_name, force=force)
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
            logger.info(f"Copying file: {dst_file} -> {src_file}")


def _delete_file(file_path, dst):
    with open(file_path, "r", encoding="utf-8") as f:
        deleted_files = f.readlines()
        for deleted_file in deleted_files:
            deleted_file = deleted_file.strip()
            deleted_file_path = os.path.join(dst, deleted_file)
            if os.path.lexists(deleted_file_path):
                os.remove(deleted_file_path)
                logger.info(f"Deleting file: {deleted_file_path}")
            else:
                logger.warning(f"File not found for deletion: {deleted_file_path}")


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

            logger.info(f"Creating symbolic link: {dst_file} -> {src_file}")


def init_submodule(main_path, dst, submodule_name, force=False):
    if os.path.lexists(dst) and len(os.listdir(dst)) > 0 and not force:
        logger.info(f"Skipping {submodule_name} initialization, as it already lexists.")
        return
    logger.info(f"Initializing submodule {submodule_name}...")
    logger.warning(
        "When you perform unpatch, the specified submodule will be fully restored to its initial state, regardless of any modifications you may have made within the submodule."
    )
    repo = Repo(main_path)
    submodule = repo.submodule(submodule_name)
    try:
        git_modules_path = os.path.join(main_path, ".git", "modules", submodule_name)
        if os.path.exists(git_modules_path):
            shutil.rmtree(git_modules_path)
        submodule_worktree_path = os.path.join(main_path, submodule_name)
        if os.path.exists(submodule_worktree_path):
            shutil.rmtree(submodule_worktree_path)
        submodule.update(init=True, force=force)
    except:
        logger.info("Retrying to initialize submodule...")
        git_modules_path = os.path.join(main_path, ".git", "modules", submodule_name)
        if os.path.exists(git_modules_path):
            shutil.rmtree(git_modules_path)
        submodule_worktree_path = os.path.join(main_path, submodule_name)
        if os.path.exists(submodule_worktree_path):
            shutil.rmtree(submodule_worktree_path)
        submodule.update(init=True, force=force)
    logger.info(f"Initialized {submodule_name} submodule.")


def commit_to_checkout(main_path, device_type=None, tasks=None, backends=None, commit=None):
    if commit:
        return commit

    newest_flagscale_commit = None
    main_repo = Repo(main_path)
    if device_type and tasks:
        # Check if device_type is in the format xxx_yyy
        if device_type.count("_") != 1 or len(device_type.split("_")) != 2:
            raise ValueError("Invalid format. Device type must be in the format xxx_yyy.")

        assert backends
        history_yaml = os.path.join(main_path, "hardware", "patch_history.yaml")
        if not os.path.exists(history_yaml):
            raise ValueError(f"Yaml {history_yaml} does not exist.")

        # Backend key
        backends_key = "+".join(sorted(backends))
        # Newest flagscale commit to checkout and unpatch
        newest_flagscale_commit = None
        # Find newest flagscale commit
        with open(history_yaml, 'r') as f:
            history = yaml.safe_load(f)
            if device_type not in history:
                raise ValueError(f"Device type {device_type} not found in {history_yaml}.")

            # Find the newest flagscale commit in the history
            for task in tasks:
                if task not in history[device_type]:
                    continue
                if backends_key not in history[device_type][task]:
                    continue
                if (
                    not isinstance(history[device_type][task][backends_key], list)
                    or not history[device_type][task][backends_key]
                ):
                    continue
                newest_flagscale_commit = history[device_type][task][backends_key][-1]
                try:
                    main_repo.commit(newest_flagscale_commit)
                    break
                except ValueError:
                    raise ValueError(
                        f"The commit ID {newest_flagscale_commit} does not exist in the FlagScale. Please check the {history_yaml}"
                    )
                    newest_flagscale_commit = None
        assert (
            newest_flagscale_commit is not None
        ), f"FlagScale Commit for device type {device_type}, task {task} is not found. Please check the {history_yaml}."
    return newest_flagscale_commit


def apply_hardware_patch(device_type, backends, commit, main_path, init_submodule, key_path=None):
    build_path = os.path.join(main_path, "build", device_type)
    final_path = os.path.join(build_path, os.path.basename(main_path))

    try:
        # Remove existing build directory if present.
        if os.path.exists(build_path):
            logger.info(f"Removing existing build path: {build_path}")
            shutil.rmtree(build_path)

        temp_path = tempfile.mkdtemp()
        logger.info(f"Step 1: Copying {main_path} to temp path {temp_path}")
        shutil.copytree(main_path, temp_path, dirs_exist_ok=True)

        repo = Repo(temp_path)
        # Stash firstly to prevent checkout failed
        repo.git.stash("push", "--include-untracked")
        logger.info(f"Step 2: Checking out {commit} in temp path {temp_path}")
        repo.git.checkout(commit)

        # Check device path
        device_path = os.path.join(temp_path, "hardware", device_type)
        if not os.path.exists(device_path):
            raise ValueError(f"{device_path} is not found.")

        # Check backend path and patch file path
        all_base_commit_id = set()
        patch_files = []
        for backend in backends:
            backend_path = os.path.join(device_path, backend)
            if not os.path.exists(backend_path):
                raise ValueError(f"{backend_path} is not found.")

            error = f"Patch files in {backend_path} must be a file with a .patch suffix and a file with a .yaml suffix."
            if len(os.listdir(backend_path)) != 2:
                raise ValueError(error)
            patch_file = None
            base_commit_id = None
            for file in os.listdir(backend_path):
                if file.endswith(".patch") or file.endswith(".encrypted"):
                    patch_file = os.path.join(backend_path, file)
                    base_commit_id = file.split(".")[0]
                    try:
                        repo.commit(base_commit_id)
                    except ValueError:
                        raise ValueError(
                            f"The commit ID {base_commit_id} does not exist in the FlagScale."
                        )
            assert patch_file
            assert base_commit_id
            all_base_commit_id.add(base_commit_id)
            patch_files.append(patch_file)
        all_base_commit_id = list(all_base_commit_id)

        # Sort the commit by appearance order
        position = {}
        rev_list = repo.git.rev_list('--topo-order', 'HEAD').splitlines()
        rev_list = [commit[:7] for commit in rev_list]
        for idx, commit in enumerate(rev_list):
            if commit in all_base_commit_id:
                position[commit] = idx

        # Check if all commits were found
        missing = set(all_base_commit_id) - set(position.keys())
        if missing:
            raise ValueError(f"The following commits were not found in rev-list: {missing}")

        sorted_commits = sorted(all_base_commit_id, key=lambda x: position[x])
        # Get the neweset base_commit_id
        base_commit_id = sorted_commits[-1]
        logger.info(f"Step 3: Finding the newset base commit {base_commit_id} to checkout.")

        temp_unpatch_path = tempfile.mkdtemp()
        logger.info(f"Step 4: Copying {temp_path} to temp unpatch path {temp_unpatch_path}")
        shutil.copytree(temp_path, temp_unpatch_path, dirs_exist_ok=True)
        repo = Repo(temp_unpatch_path)
        repo.git.checkout(base_commit_id)

        logger.info(f"Step 5: Applying patch:")
        for patch_file in patch_files:
            # Check if the patch file is encrypted
            new_patch_file = patch_file
            if patch_file.endswith(".encrypted"):
                if key_path is not None:
                    private_key_path = os.path.join(key_path, "private_key.pem")
                    new_patch_file = decrypt_file(patch_file, private_key_path)
                else:
                    raise ValueError(
                        f"Patch file {patch_file} is encrypted, but no key path provided."
                    )
            repo.git.apply("--index", "--whitespace", "fix", new_patch_file)
            logger.info(f"Patch {new_patch_file} has been applied.")

        logger.info(f"Step 6: Initializing submodule in temp unpatch path {temp_unpatch_path}...")
        if init_submodule:
            for backend in backends:
                submodule_name = f"third_party/{backend}"
                dst = os.path.join(temp_unpatch_path, "third_party", backend)
                src = os.path.join(temp_unpatch_path, "flagscale", "backends", backend)
                # NOTE: mode must be 'copy' because the temp unpatch path will be moved
                unpatch(temp_unpatch_path, src, dst, submodule_name, mode="copy", force=True)

        logger.info(f"Step 7: Moving patched temp path {temp_unpatch_path} to {final_path}")
        os.makedirs(build_path, exist_ok=True)
        shutil.move(temp_unpatch_path, final_path)
        logger.info(f"Unpatch Ended.")

    except Exception as e:
        logger.error(f"Exception occurred: {e}", exc_info=True)

        # Clean up temp directory
        if "temp_path" in locals() and os.path.exists(temp_path):
            logger.info(f"Cleaning up temp path: {temp_path}")
            shutil.rmtree(temp_path, ignore_errors=True)

        # Clean up temp directory
        if "temp_unpatch_path" in locals() and os.path.exists(temp_unpatch_path):
            logger.info(f"Cleaning up temp path: {temp_unpatch_path}")
            shutil.rmtree(temp_unpatch_path, ignore_errors=True)

        # Clean up build directory
        if os.path.exists(build_path):
            logger.info(f"Cleaning up build path: {build_path}")
            shutil.rmtree(build_path, ignore_errors=True)

        raise ValueError("Error occurred during unpatching.")
    return final_path


def decrypt_file(encrypted_file_name, private_key_path):
    with open(private_key_path, "rb") as private_file:
        private_key = serialization.load_pem_private_key(
            private_file.read(), password=None, backend=default_backend()
        )
    with open(encrypted_file_name, "rb") as encrypted_file:
        print(encrypted_file_name, encrypted_file)
        iv = encrypted_file.read(16)
        encrypted_aes_key = encrypted_file.read(512)
        encrypted_data = encrypted_file.read()

    aes_key = None
    try:
        aes_key = private_key.decrypt(
            encrypted_aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None
            ),
        )
        print("AES key decrypted successfully.")
    except Exception as e:
        print(f"Decryption of AES key failed: {e}")
        return  # Exit the function if decryption fails
    if aes_key is None:
        print("AES key could not be decrypted.")
        return
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    try:
        decrypted_padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
    except Exception as e:
        print(f"Decryption of data failed: {e}")
        return
    padding_length = decrypted_padded_data[-1]
    decrypted_data = decrypted_padded_data[:-padding_length]
    decrypted_file_path = os.path.splitext(encrypted_file_name)[0]
    with open(decrypted_file_path, "wb") as decrypted_file:
        decrypted_file_path.write(decrypted_data)

    return decrypted_file_path


def validate_args(device_type, task, commit, main_path):
    main_repo = Repo(main_path)
    if commit:
        # Check if the commit exists in the FlagScale
        try:
            main_repo.commit(commit)
        except ValueError:
            raise ValueError(f"Commit {commit} does not exist in the FlagScale.")
    if device_type:
        if (
            device_type.count("_") != 1
            or len(device_type.split("_")) != 2
            or not device_type.split("_")[0][0].isupper()
        ):
            raise ValueError("Device type is not invalid!")

    if device_type or task:
        assert device_type and task, "The args device_type, task must not be None."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch or unpatch backend with symlinks.")
    parser.add_argument(
        "--backend",
        nargs="+",
        choices=["Megatron-LM", "vllm", "Megatron-Energon", "FlagScale"],
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
        "--device-type", type=str, default=None, help="Device type. Default is None."
    )
    parser.add_argument(
        "--task",
        nargs="+",
        default=None,
        choices=["train", "inference", "post_train"],
        help="Task. Default is None",
    )
    parser.add_argument(
        "--commit", type=str, default=None, help="Unpatch based on this commit. Default is None."
    )
    parser.add_argument(
        "--force", action="store_true", help="Force update the backend even if it already exists."
    )
    parser.add_argument(
        "--no-init-submodule",
        action="store_false",
        dest="init_submodule",
        help="Do not initialize and update submodules. Default is True.",
    )
    parser.add_argument(
        "--key-path",
        type=str,
        default=None,
        help="The path for storing public and private keys. Be careful not to upload to the Git repository.",
    )

    args = parser.parse_args()
    backends = args.backend
    device_type = args.device_type
    tasks = args.task
    commit = args.commit
    key_path = args.key_path

    # When unpatching, the submodule will be initialized forcefully
    args.force = True

    if not isinstance(backends, list):
        backends = [backends]

    if tasks is not None and not isinstance(tasks, list):
        tasks = [tasks]

    # FlagScale/tools/patch
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # FlagScale/tools
    script_dir = os.path.dirname(script_dir)
    # FlagScale
    main_path = os.path.dirname(script_dir)

    validate_args(device_type, tasks, commit, main_path)

    if FLAGSCALE_BACKEND in backends:
        assert (
            device_type is not None
        ), "FlagScale unpatch only can be applied with hardware unpatch."

    # Check patch exist
    commit = commit_to_checkout(main_path, device_type, tasks, backends, commit)
    if commit is not None:
        # Checkout to the commit and apply the patch to build FlagScale
        apply_hardware_patch(
            device_type, backends, commit, main_path, args.init_submodule, key_path=key_path
        )

    else:
        for backend in backends:
            submodule_name = f"third_party/{backend}"
            dst = os.path.join(main_path, "third_party", backend)
            src = os.path.join(main_path, "flagscale", "backends", backend)
            unpatch(main_path, src, dst, submodule_name, mode=args.mode, force=args.force)
