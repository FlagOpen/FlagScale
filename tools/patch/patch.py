import argparse
import copy
import logging
import os
import shutil
import sys
import tempfile

import yaml
from git.repo import Repo

DELETED_FILE_NAME = "deleted_files.txt"
FLAGSCALE_BACKEND = "FlagScale"


logger = logging.getLogger("FlagScalePatchLogger")
logger.setLevel(logging.INFO)


if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[FlagScale-Patch] %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


def patch(
    main_path,
    submodule_name,
    src,
    dst,
    mode="symlink",
    base_commit_id=None,
    backends=None,
):
    """
    Sync the submodule modifications to the corresponding backend in FlagScale.
    """
    if submodule_name.split("/")[1] != FLAGSCALE_BACKEND:
        logger.info(f"Patching backend {submodule_name}...")
        main_repo = Repo(main_path)
        submodule = main_repo.submodule(submodule_name)
        sub_repo = submodule.module()
        base_commit_hash = submodule.hexsha
        logger.info(
            f"Base commit hash of submodule {submodule_name} is {base_commit_hash}."
        )

        # Get submodule commit tree
        base_commit = sub_repo.commit(base_commit_hash)
        base_tree = base_commit.tree

        index = sub_repo.index
        index_tree_hash = index.write_tree()
        file_statuses = {}

        # Get diff with base commit
        diff_index = base_tree.diff(index_tree_hash)
        # Process the diff between the staged and the base commit
        for diff in diff_index:
            if diff.new_file:
                status = "A"
                file_path = diff.b_path
                file_statuses[file_path] = [status]
            elif diff.deleted_file:
                status = "D"
                file_path = diff.a_path
                file_statuses[file_path] = [status]
            elif diff.renamed_file:
                status = "R"
                file_path = diff.b_path
                file_statuses[diff.a_path] = [status, file_path]
            elif diff.change_type == "M":
                status = "M"
                assert diff.a_path == diff.b_path
                file_path = diff.b_path
                file_statuses[file_path] = [status]
            elif diff.change_type == "T":
                status = "T"
                assert diff.a_path == diff.b_path
                file_path = diff.b_path
                file_statuses[file_path] = [status]
            elif diff.change_type == "U":
                raise ValueError(f"Unmerged status is not supported.")
            else:
                raise ValueError(f"Unsupported  status: {diff.change_type}.")

        # Get diff with working directory
        diff_workdir = index.diff(None)
        # Process the diff between the working directory and the staged
        for diff in diff_workdir:
            if diff.new_file:
                status = "A"
                file_path = diff.b_path
                file_statuses[file_path] = [status]
            elif diff.deleted_file:
                status = "D"
                file_path = diff.a_path
                file_statuses[file_path] = [status]
            elif diff.renamed_file:
                status = "R"
                file_path = diff.b_path
                file_statuses[diff.a_path] = [status, file_path]
            elif diff.change_type == "M":
                status = "M"
                assert diff.a_path == diff.b_path
                file_path = diff.b_path
                file_statuses[file_path] = [status]
            elif diff.change_type == "T":
                status = "T"
                assert diff.a_path == diff.b_path
                file_path = diff.b_path
                file_statuses[file_path] = [status]
            elif diff.change_type == "U":
                raise ValueError(f"Unmerged status is not supported.")
            else:
                raise ValueError(f"Unsupported  status: {diff.change_type}.")
            file_statuses[file_path] = status

        # Get untracked files
        untracked_files = sub_repo.untracked_files
        for file in untracked_files:
            file_statuses[file] = ["UT"]

        # The file status may be overwritten, so we follow the sequence of staged, working dir, untracked.

        file_status_deleted = {}
        temp_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        for file_path in file_statuses:
            if file_statuses[file_path][0] == "D":
                file_status_deleted[file_path] = file_statuses[file_path]

        for file_path in file_statuses:
            if file_statuses[file_path][0] == "D":
                continue
            _sync(file_path, file_statuses[file_path], src, dst, temp_file, mode=mode)

        # Process the deleted files
        if file_status_deleted:
            try:
                for file_path in file_status_deleted:
                    assert file_statuses[file_path][0] == "D"
                    _sync(
                        file_path,
                        file_status_deleted[file_path],
                        src,
                        dst,
                        temp_file,
                        mode=mode,
                    )
                deleted_log = os.path.join(src, DELETED_FILE_NAME)
                temp_file.close()

                shutil.move(temp_file.name, deleted_log)
                if os.path.lexists(temp_file.name):
                    os.remove(temp_file.name)

            except Exception as e:
                print(f"Error occurred while processing deleted files: {e}")
                temp_file.close()
                if os.path.lexists(temp_file.name):
                    os.remove(temp_file.name)
                raise e

    if base_commit_id:
        patch_info = prompt_info(main_path, backends)
        generate_patch_file(main_path, base_commit_id, patch_info)


def prompt_info(main_path, backends):
    logger.info("Prompting for patch information: ")
    # 1. tasks (support multiple comma-separated)
    valid_tasks = {"train", "inference", "post_train"}
    while True:
        task_input = input(
            "1. Please enter task (valid task train/inference/post_train) (separated with commas, e.g., train, post_train): "
        ).strip()
        task_list = [t.strip() for t in task_input.split(",") if t.strip()]
        if all(t in valid_tasks for t in task_list):
            break
        logger.info(f"Invalid task(s). Each must be one of: {', '.join(valid_tasks)}")
    # 2. backends
    third_party_dir = os.path.join(main_path, "third_party")
    available_submodules = [
        d
        for d in os.listdir(third_party_dir)
        if os.path.isdir(os.path.join(third_party_dir, d))
    ]
    available_submodules.append("FlagScale")
    assert isinstance(backends, list)
    invalid = [b for b in backends if b not in available_submodules]
    if invalid:
        logger.error(f"Backends are not valid submodules: {', '.join(invalid)}", exc_info=True)
        raise ValueError(
            f"Available submodules: {', '.join(sorted(available_submodules))}"
        )

    backends_version = {}
    print("2. Please enter backends version: ")
    for backend in backends:
        version = input(f"    {backend} version: ").strip()
        while not version:
            logger.info(f"Version for {backend} cannot be empty.")
            version = input(f"    {backend} version: ").strip()
        backends_version[backend] = version

    # 3. device type
    device_type = input("3. Please enter device type (e.g., HARDWARE_CHIP): ").strip()
    while device_type.count("_") != 1 or len(device_type.split("_")) != 2:
        logger.info("Invalid format. Device type must be in the format xxx_yyy.")
        device_type = input(
            "3. Please enter device type (e.g., HARDWARE_CHIP): "
        ).strip()

    # 4. FlagScale-compatible models
    model_input = input(
        "4. Please enter flagScale-compatible models (separated with commas): "
    ).strip()
    models = [m.strip() for m in model_input.split(",") if m.strip()]
    while not models:
        logger.info("At least one FlagScale-compatible model must be provided.")
        model_input = input(
            "4. Please enter FlagScale-compatible models (separated with commas): "
        ).strip()
        models = [m.strip() for m in model_input.split(",") if m.strip()]

    # 5. Commit message
    commit_msg = input("5. Please enter commit message: ").strip()
    while not commit_msg:
        logger.info("Commit message cannot be empty.")
        commit_msg = input("5. Please enter commit message: ").strip()

    # 6. Contact (optional)
    contact_prompt = "6. Please enter email (optional): "
    contact = input(contact_prompt).strip()

    return {
        "task": task_list,
        "backends_version": backends_version,
        "device_type": device_type,
        "models": models,
        "contact": contact,
        "commit_msg": commit_msg,
    }


def generate_patch_file(main_path: str, base_commit_id: str, patch_info: dict):
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

        logger.info(
            "Generating the patch file:"
        )
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

        # Step3: Commit with message "Patch for {base_commit_id}".
        logger.info("Step 3: Committing changes on the temp_branch_for_hardware_patch...")
        repo.git.commit("-m", f"Patch for {base_commit_id}")

        # Step4: Generate patch diff between base_commit_id and HEAD, writing into temp_file.
        logger.info(f"Step 4: Generating diff patch from {base_commit_id} to HEAD...")
        backends = copy.deepcopy(list(patch_info["backends_version"].keys()))
        patches = {}

        # Diff excludes the submodules
        flagscale_diff_args = [
            base_commit_id,
            "HEAD",
            "--binary",
            "--ignore-submodules=all",
            "--",
        ]

        tmep_patch_files = []
        # Generate patch for each backend
        for backend in backends:
            if backend == FLAGSCALE_BACKEND:
                continue
            backend_dir = os.path.join(main_path, "flagscale", "backends", backend)
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, mode="w", encoding="utf-8"
            )
            temp_patch_path = temp_file.name
            tmep_patch_files.append(temp_patch_path)
            diff_data = repo.git.diff(
                base_commit_id,
                "HEAD",
                "--binary",
                "--ignore-submodules=all",
                "--",
                f"{backend_dir}",
            )
            if not diff_data:
                raise ValueError(f"No changes in backend {backend}.")

            temp_file.write(diff_data)
            # add \n to the end of the file
            temp_file.write("\n")
            temp_file.flush()
            flagscale_diff_args.append(f':(exclude){backend_dir}')

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
                main_path, "hardware", patch_info["device_type"], backend
            )
            patches[backend] = [patch_dir, temp_patch_path, temp_yaml_path]

        # Generate patch for FlagScale
        if FLAGSCALE_BACKEND in backends:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, mode="w", encoding="utf-8"
            )
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

        # Step5: Checkout the base_commit_id commit.
        logger.info(f"Step 5: Checking out {base_commit_id}...")
        repo.git.checkout(base_commit_id)

        # Step6: Stage the patch file.
        logger.info("Step 6: Staging the generated patch file...")
        file_name = f"{base_commit_id[:7]}.patch"
        yaml_file_name = f"{base_commit_id[:7]}.yaml"
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
        logger.info(
            "Step 8: Deleting temporary branch 'temp_branch_for_hardware_patch' locally..."
        )
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


def _sync(file_path, status, src, dst, f=None, mode="symlink"):
    src_file_path = os.path.join(src, file_path)
    dst_file_path = os.path.join(dst, file_path)
    change_type = status[0]

    symbolic_error = "Defining symbolic links in the submodule is not supported except for those defined in FlagScale"
    typechange_error = "File type changes are not supported in the submodule"
    if change_type == "T":
        is_symlink = os.path.islink(dst_file_path)
        if is_symlink:
            if not os.path.lexists(src_file_path):
                raise ValueError(f"{symbolic_error}: {dst_file_path}")
        else:
            raise ValueError(f"{typechange_error}: {dst_file_path}")

    elif change_type in ["A", "UT"]:
        is_symlink = os.path.islink(dst_file_path)
        if is_symlink:
            if not os.path.lexists(src_file_path):
                real_path = os.readlink(dst_file_path)
                if os.path.lexists(real_path):
                    os.makedirs(os.path.dirname(src_file_path), exist_ok=True)
                    shutil.move(real_path, src_file_path)
                    logger.info(
                        f"Move {real_path} to {src_file_path} and create symbolic link {dst_file_path} -> {src_file_path}"
                    )
                    if os.path.lexists(dst_file_path):
                        os.remove(dst_file_path)
                    os.symlink(src_file_path, dst_file_path)
                else:
                    raise ValueError(f"{symbolic_error}: {dst_file_path}")
        else:
            _create_file(src_file_path, dst_file_path, mode=mode)

    elif change_type == "D":
        if os.path.lexists(src_file_path):
            os.remove(src_file_path)
            logger.debug(f"File {src_file_path} has been deleted.")
        else:
            assert f
            f.write(f"{file_path}\n")
            f.flush()

    elif change_type == "M":
        is_symlink = os.path.islink(dst_file_path)
        if is_symlink:
            raise ValueError(
                "Modified symbolic links in the submodule is not supported except for those defined in FlagScale"
            )
        _create_file(src_file_path, dst_file_path, mode=mode)

    elif change_type == "R":
        assert len(status) == 2
        rel_dst_path = status[1]
        renamed_dst_file_path = os.path.join(dst, rel_dst_path)
        is_symlink = os.path.islink(renamed_dst_file_path)
        renamed_src_file_path = os.path.join(src, rel_dst_path)
        if is_symlink:
            real_path = os.readlink(renamed_dst_file_path)
            os.makedirs(os.path.dirname(renamed_src_file_path), exist_ok=True)
            if real_path != renamed_src_file_path:
                shutil.move(real_path, renamed_src_file_path)
                logger.info(
                    f"Move {real_path} to {renamed_src_file_path} and create symbolic link {renamed_dst_file_path} -> {renamed_src_file_path}"
                )
            if os.path.lexists(renamed_dst_file_path):
                os.remove(renamed_dst_file_path)
            os.symlink(renamed_src_file_path, renamed_dst_file_path)
        else:
            assert not os.path.lexists(renamed_src_file_path)
            _create_file(renamed_src_file_path, renamed_dst_file_path, mode=mode)
            assert f
            f.write(f"{file_path}\n")
            f.flush()


def _create_file(source_file, target_file, mode="symlink"):
    if os.path.lexists(source_file):
        logger.warning(f"File {source_file} will be covered by {target_file}.")
    assert os.path.lexists(target_file)

    source_dir = os.path.dirname(source_file)
    if not os.path.lexists(source_dir):
        os.makedirs(source_dir, exist_ok=True)

    shutil.copyfile(target_file, source_file)
    if mode == "symlink":
        if os.path.lexists(target_file):
            os.remove(target_file)
        os.symlink(source_file, target_file)
        logger.info(
            f"File {target_file} has been copied to {source_file} and Create symbolic link {target_file} -> {source_file}."
        )
    elif mode == "copy":
        logger.info(f"File {source_file} has been copied to {target_file}.")
    else:
        raise ValueError(f"Unsupported mode: {mode}.")


def validate_base_commit(base_commit_id, main_path):
    main_repo = Repo(main_path)
    if base_commit_id:
        # Check if the base_commit_id exists in the FlagScale
        try:
            main_repo.commit(base_commit_id)
        except ValueError:
            raise ValueError(
                f"The commit ID {base_commit_id} does not exist in the FlagScale."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync submodule modifications to the corresponding backend in FlagScale."
    )
    parser.add_argument(
        "--backend",
        nargs="+",
        choices=["Megatron-LM", "vllm", "Megatron-Energon", "FlagScale"],
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
        "--base-commit-id",
        type=str,
        default=None,
        help="Base commit ID to reference in the repo. Default is None.",
    )

    args = parser.parse_args()
    backends = args.backend
    mode = args.mode
    base_commit_id = args.base_commit_id
    if not isinstance(backends, list):
        backends = [backends]
    # FlagScale/tools/patch
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # FlagScale/tools
    script_dir = os.path.dirname(script_dir)
    # FlagScale
    main_path = os.path.dirname(script_dir)
    validate_base_commit(args.base_commit_id, main_path)

    if FLAGSCALE_BACKEND in backends:
        assert base_commit_id is not None, "FlagScale patch only can be generated with hardware."

    multi_backends = len(backends) > 1
    if multi_backends and base_commit_id:
        for backend in backends:
            submodule_name = f"third_party/{backend}"
            dst = os.path.join(main_path, "third_party", backend)
            src = os.path.join(main_path, "flagscale", "backends", backend)
            patch(main_path, submodule_name, src, dst, mode)
        patch_info = prompt_info(main_path, backends)
        generate_patch_file(main_path, base_commit_id, patch_info)

    else:
        for backend in backends:
            submodule_name = f"third_party/{backend}"
            dst = os.path.join(main_path, "third_party", backend)
            src = os.path.join(main_path, "flagscale", "backends", backend)
            patch(main_path, submodule_name, src, dst, mode, base_commit_id, backends)
