import argparse
import copy
import os
import shutil
import sys
import tempfile

import yaml

from encryption_utils import encrypt_file, generate_rsa_keypair
from file_utils import sync_to_flagscale
from git.repo import Repo
from git_utils import (
    get_diff_between_commit_and_now,
    get_file_statuses_for_staged_or_unstaged,
    get_file_statuses_for_untracked,
    check_git_user_info,
)
from logger_utils import get_patch_logger

DELETED_FILE_NAME = "deleted_files.txt"
FLAGSCALE_BACKEND = "FlagScale"
logger = get_patch_logger()


def patch(main_path, submodule_name, src, dst, mode="symlink", **kwargs):
    """
    Sync the submodule modifications to the corresponding backend in FlagScale.
    Args:
        main_path (str): The path to the repository.
        submodule_name (str): The name of the submodule to be patched, e.g., "Mgeatron-LM".
        src (str): The source directory of the submodule, e.g., "flagscale/backends/Megatron-LM".
        dst (str): The destination directory of the submodule, e.g., "third_party/Megatron-LM".
        mode (str): The mode to patch (default: symlink),
                    it means that the file will be copied to the source and a symbolic link from src to dst will be created.
                    If the mode is copy, the file will be copied to the source and the symbolic link will not be created.
    """

    """
    These arguments are used for hardware patch.
    Args:
        commit (str): The commit hash based to patch (default: None).
        backends (list): List of backends to patch (default: None).
        device_type (str): The device type (default: None).
        tasks (list): List of tasks to patch (default: None).
        key_path (str): The path for public and private keys (default: None).
    """
    commit = kwargs.get('commit', None)
    backends = kwargs.get('backends', None)
    device_type = kwargs.get('device_type', None)
    tasks = kwargs.get('tasks', None)
    key_path = kwargs.get('key_path', None)

    # For hardware patch, FlagScale is a submodule of the main repo.
    # But for users, they don't need to know this.
    if submodule_name != FLAGSCALE_BACKEND:
        submodule_path = "third_party" + "/" + submodule_name
        logger.info(f"Patching backend {submodule_path}...")

        # Get the submodule repo and the commit in FlagScale.
        main_repo = Repo(main_path)
        submodule = main_repo.submodule(submodule_path)
        sub_repo = submodule.module()
        submodule_commit_in_fs = submodule.hexsha
        logger.info(f"Base commit hash of submodule {submodule_path} is {submodule_commit_in_fs}.")

        # Get all differences between the submodule specified commit and now.
        # The differences include staged, working directory, and untracked files.
        staged_diff, unstaged_diff, untracked_files = get_diff_between_commit_and_now(
            sub_repo, submodule_commit_in_fs
        )

        file_statuses = {}
        # Process the diff between the staged and the base commit
        staged_file_statuses = get_file_statuses_for_staged_or_unstaged(staged_diff)
        file_statuses.update(staged_file_statuses)
        # Process the diff between the working directory and the staged
        unstaged_file_statuses = get_file_statuses_for_staged_or_unstaged(unstaged_diff)
        file_statuses.update(unstaged_file_statuses)
        # Process the untracked files
        untracked_file_statuses = get_file_statuses_for_untracked(untracked_files)
        file_statuses.update(untracked_file_statuses)

        # Process the deleted files
        file_status_deleted = {}
        for file_path in file_statuses:
            if file_statuses[file_path][0] == "D":
                file_status_deleted[file_path] = file_statuses[file_path]

        # Sync the files to FlagScale and skip the deleted files firstly
        # Temp file is used to store the deleted files
        temp_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        for file_path in file_statuses:
            if file_statuses[file_path][0] == "D":
                continue
            sync_to_flagscale(file_path, file_statuses[file_path], src, dst, temp_file, mode=mode)

        # Process the deleted files
        if file_status_deleted:
            try:
                for file_path in file_status_deleted:
                    assert file_statuses[file_path][0] == "D"
                    sync_to_flagscale(
                        file_path, file_status_deleted[file_path], src, dst, temp_file, mode=mode
                    )
                deleted_log = os.path.join(src, DELETED_FILE_NAME)
                temp_file.close()

                shutil.move(temp_file.name, deleted_log)
                if os.path.lexists(temp_file.name):
                    os.remove(temp_file.name)

            except Exception as e:
                print(f"Error occurred while processing deleted files: {e}")
                # Rollback
                temp_file.close()
                if os.path.lexists(temp_file.name):
                    os.remove(temp_file.name)
                raise e

    # For hardware patch, the commit hash is specified.
    if commit:
        patch_info = prompt_info(main_path, backends, device_type, tasks)
        generate_patch_file(main_path, commit, patch_info, key_path=key_path)


def prompt_info(main_path, backends, device_type, tasks):
    logger.info("Prompting for patch information: ")

    backends_version = {}
    print("1. Please enter backends version: ")
    for backend in backends:
        version = input(f"    {backend} version: ").strip()
        while not version:
            logger.info(f"Version for {backend} cannot be empty.")
            version = input(f"    {backend} version: ").strip()
        backends_version[backend] = version

    # FlagScale-compatible models
    model_input = input(
        "2. Please enter flagScale-compatible models (separated with commas): "
    ).strip()
    models = [m.strip() for m in model_input.split(",") if m.strip()]
    while not models:
        logger.info("At least one FlagScale-compatible model must be provided.")
        model_input = input(
            "2. Please enter FlagScale-compatible models (separated with commas): "
        ).strip()
        models = [m.strip() for m in model_input.split(",") if m.strip()]

    # 3. Commit message
    commit_msg = input("3. Please enter commit message: ").strip()
    while not commit_msg:
        logger.info("Commit message cannot be empty.")
        commit_msg = input("3. Please enter commit message: ").strip()

    # 4. Contact (optional)
    contact_prompt = "4. Please enter email (optional): "
    contact = input(contact_prompt).strip()

    return {
        "task": tasks,
        "backends_version": backends_version,
        "device_type": device_type,
        "models": models,
        "contact": contact,
        "commit_msg": commit_msg,
    }


def generate_patch_file(main_path: str, commit: str, patch_info: dict, key_path=None):
    repo = Repo(main_path)
    assert not repo.bare

    """
    This function performs the following steps.
    """

    try:
        # Initialize the repository
        repo = Repo(main_path)
        current_branch = repo.active_branch.name
        if repo.bare:
            raise Exception("Repository is bare. Cannot proceed.")

        logger.info("Generating the patch file:")
        # Step 1: Stash all, including untracked, and create temp_branch_for_hardware_patch
        logger.info(
            "Step 1: Stashing current changes (including untracked) and create the temp_branch_for_hardware_patch..."
        )
        temp_branch = "temp_branch_for_hardware_patch"
        repo.git.stash("push", "--include-untracked")
        stash_pop = False
        if temp_branch in repo.heads:
            logger.info(
                "Temporary branch 'temp_branch_for_hardware_patch' already exists, deleting..."
            )
            repo.git.branch("-D", temp_branch)
        repo.git.checkout("-b", temp_branch)

        # Step2: Apply stash on the temp_branch_for_hardware_patch and add
        logger.info(
            "Step 2: Applying stashed changes and add changes on the temp_branch_for_hardware_patch..."
        )
        repo.git.stash("apply")
        repo.git.add(all=True)

        # Step3: Commit with message "Patch for {commit}".
        logger.info("Step 3: Committing changes on the temp_branch_for_hardware_patch...")
        repo.git.commit("--no-verify", "-m", f"Patch for {commit}")

        # Step4: Generate patch diff between commit and HEAD, writing into temp_file.
        logger.info(f"Step 4: Generating diff patch from {commit} to HEAD...")
        backends = copy.deepcopy(list(patch_info["backends_version"].keys()))
        patches = {}

        # Diff excludes the submodules
        flagscale_diff_args = [commit, "HEAD", "--binary", "--ignore-submodules=all", "--"]

        tmep_patch_files = []
        # Generate patch for each backend
        for backend in backends:
            if backend == FLAGSCALE_BACKEND:
                continue
            backend_dir = os.path.join(main_path, "flagscale", "backends", backend)
            temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
            temp_patch_path = temp_file.name
            tmep_patch_files.append(temp_patch_path)
            diff_data = repo.git.diff(
                commit, "HEAD", "--binary", "--ignore-submodules=all", "--", f"{backend_dir}"
            )
            if not diff_data:
                raise ValueError(f"No changes in backend {backend}.")

            temp_file.write(diff_data)
            # add \n to the end of the file
            temp_file.write("\n")
            temp_file.flush()
            if key_path is not None:
                temp_patch_path = encrypt_file(temp_patch_path, key_path)
                logger.info(f"Encrypted patch file {temp_patch_path} with public key.")

            flagscale_diff_args.append(f':(exclude){backend_dir}')

            temp_yaml_file = tempfile.NamedTemporaryFile(
                delete=False, mode="w", encoding="utf-8", suffix=".yaml"
            )
            temp_yaml_path = temp_yaml_file.name
            tmep_patch_files.append(temp_yaml_path)
            data = copy.deepcopy(patch_info)
            data["commit"] = commit
            del data["commit_msg"]
            yaml.dump(data, temp_yaml_file, sort_keys=True, allow_unicode=True)
            temp_yaml_file.flush()
            patch_dir = os.path.join(main_path, "hardware", patch_info["device_type"], backend)
            patches[backend] = [patch_dir, temp_patch_path, temp_yaml_path]

        # Generate patch for FlagScale
        if FLAGSCALE_BACKEND in backends:
            temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
            temp_patch_path = temp_file.name
            tmep_patch_files.append(temp_patch_path)
            diff_data = repo.git.diff(*flagscale_diff_args)
            if not diff_data:
                raise ValueError(f"No changes in backend {FLAGSCALE_BACKEND}.")
            else:
                temp_file.write(diff_data)
                # add \n to the end of the file
                temp_file.write("\n")
                temp_file.flush()
                if key_path is not None:
                    temp_patch_path = encrypt_file(temp_patch_path, key_path)
                    logger.info(f"Encrypted patch file {temp_patch_path} with public key.")

                temp_yaml_file = tempfile.NamedTemporaryFile(
                    delete=False, mode="w", encoding="utf-8", suffix=".yaml"
                )
                temp_yaml_path = temp_yaml_file.name
                tmep_patch_files.append(temp_yaml_path)
                data = copy.deepcopy(patch_info)
                del data["commit_msg"]
                yaml.dump(data, temp_yaml_file, sort_keys=True, allow_unicode=True)
                temp_yaml_file.flush()
                patch_dir = os.path.join(
                    main_path, "hardware", patch_info["device_type"], "FlagScale"
                )
                patches["FlagScale"] = [patch_dir, temp_patch_path, temp_yaml_path]

        repo.git.checkout(current_branch)
        repo.git.stash("pop")
        stash_pop = True

        # Step5: Checkout the commit commit.
        logger.info(f"Step 5: Checking out {commit}...")
        repo.git.checkout(commit)

        # Step6: Stage the patch file.
        logger.info("Step 6: Staging the generated patch file...")
        file_name = f"diff.patch" if key_path is None else f"diff.patch.encrypted"
        yaml_file_name = f"diff.yaml"
        patch_dir_need_to_clean = []
        for backend in patches:
            patch_dir, temp_patch_path, temp_yaml_path = patches[backend]
            patch_dir_exist = os.path.exists(patch_dir)

            if not patch_dir_exist:
                patch_dir_need_to_clean.append(patch_dir)

            # Remove the backend
            if os.path.exists(patch_dir):
                shutil.rmtree(patch_dir)
            os.makedirs(patch_dir, exist_ok=True)
            print("backend temp_patch_path", backend, temp_patch_path)
            shutil.copy(temp_patch_path, os.path.join(patch_dir, file_name))
            shutil.copy(temp_yaml_path, os.path.join(patch_dir, yaml_file_name))
            repo.git.add(os.path.join(patch_dir, file_name))
            repo.git.add(os.path.join(patch_dir, yaml_file_name))

        # clean up the temp files
        for temp_patch_path in tmep_patch_files:
            if os.path.exists(temp_patch_path):
                os.remove(temp_patch_path)
                logger.debug(f"Temporary patch file {temp_patch_path} deleted.")

        # Step7: Commit the patch file with the same message.
        logger.info("Step 7: Committing the patch file...")
        commit_msg = patch_info["commit_msg"]
        repo.git.commit("-m", commit_msg)

        # Step 8: Delete the temporary branch locally.
        logger.info("Step 8: Deleting temporary branch 'temp_branch_for_hardware_patch' locally...")
        repo.git.branch("-D", temp_branch)

        logger.info(
            "Commit successfully! If you want to push, try 'git push origin HEAD:refs/heads/(your branch)' or  'git push --force origin HEAD:refs/heads/(your branch)'"
        )

    except Exception as e:
        logger.error(f"{e}", exc_info=True)
        logger.info(f"Rolling back to current branch...")
        repo.git.checkout(current_branch)
        if "stash_pop" in locals() and not stash_pop:
            repo.git.stash("pop")
            stash_pop = True

    finally:
        try:
            # Clean up the patch dir
            if "patch_dir_need_to_clean" in locals():
                for patch_dir in patch_dir_need_to_clean:
                    if os.path.exists(patch_dir):
                        shutil.rmtree(patch_dir)
                        logger.debug(f"Temporary patch dir {patch_dir} deleted.")

            # Clean up the temp files
            if "tmep_patch_files" in locals():
                for temp_patch_path in tmep_patch_files:
                    if os.path.exists(temp_patch_path):
                        os.remove(temp_patch_path)
                        logger.debug(f"Temporary patch file {temp_patch_path} deleted.")

            # Clean up the temporary branch
            if temp_branch in repo.heads:
                repo.git.branch("-D", temp_branch)

        except Exception as cleanup_error:
            logger.error(f"Failed to delete temporary: {cleanup_error}", exc_info=True)


def validate_patch_args(device_type, task, commit, main_path):
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

    if commit or device_type or task:
        assert (
            commit and device_type and task
        ), "The args commit, device_type, task must not be None."


def normalize_backend(backend):
    """
    Normalize backend to standard backend names

    Args:
        backend (str): Backend name provided by the user.

    Returns:
        str: Standardized backend name.
    """

    input_lower = backend.lower()

    if input_lower in ["megatron", "megatron-lm"]:
        return "Megatron-LM"
    elif input_lower in ["energon", "megatron-energon"]:
        return "Megatron-Energon"
    elif input_lower in ["fs", "flagscale"]:
        return "FlagScale"
    elif input_lower == "vllm":
        return "vllm"

    raise ValueError(f'Unsupported backend {backend}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync submodule modifications to the corresponding backend in FlagScale."
    )
    parser.add_argument(
        "--backend",
        nargs="+",
        type=normalize_backend,
        default=["Megatron-LM"],
        help="Backend to patch (default: Megatron-LM)",
    )

    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Mode to patch (default: symlink, it means that the file will be copied to the source and a symbolic link will be created)",
    )
    parser.add_argument(
        "--commit", type=str, default=None, help="Patch based on this commit. Default is None."
    )
    parser.add_argument(
        "--device-type", type=str, default=None, help="Device type. Default is None."
    )
    parser.add_argument("--task", nargs="+", default=None, help="Task. Default is None")
    parser.add_argument(
        "--key-path",
        type=str,
        default=None,
        help="The path for storing public and private keys. Be careful not to upload to the Git repository.",
    )

    args = parser.parse_args()
    backends = args.backend
    mode = args.mode
    commit = args.commit
    tasks = args.task
    device_type = args.device_type
    key_path = args.key_path

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

    check_git_user_info(main_path)

    validate_patch_args(device_type, tasks, commit, main_path)

    if FLAGSCALE_BACKEND in backends:
        assert commit is not None, "FlagScale patch only can be generated with hardware."

    multi_backends = len(backends) > 1
    if multi_backends and commit:
        for backend in backends:
            dst = os.path.join(main_path, "third_party", backend)
            src = os.path.join(main_path, "flagscale", "backends", backend)
            patch(main_path, backend, src, dst, mode)
        patch_info = prompt_info(main_path, backends, device_type, tasks)
        generate_patch_file(main_path, commit, patch_info, key_path=key_path)

    else:
        for backend in backends:
            dst = os.path.join(main_path, "third_party", backend)
            src = os.path.join(main_path, "flagscale", "backends", backend)
            patch(
                main_path, backend, src, dst, mode, commit=commit, backends=backends, device_type=device_type, tasks=tasks, key_path=key_path
            )
