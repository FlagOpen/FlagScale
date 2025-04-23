import argparse
import os
import shutil
import sys
import tempfile

from git.repo import Repo


def patch(main_path, submodule_name, src, dst, mode="symlink", base_commit_id=None):
    """
    Sync the submodule modifications to the corresponding backend in FlagScale.
    """
    print(f"Patching backend {submodule_name}...")
    main_repo = Repo(main_path)
    # raise ValueError(help(main_repo.submodule))
    submodule = main_repo.submodule(submodule_name)
    sub_repo = submodule.module()
    base_commit_hash = submodule.hexsha
    print(f"Base commit hash of submodule {submodule_name} is {base_commit_hash}.")

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
    print(file_statuses)

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
            deleted_log = os.path.join(src, "deleted_files.txt")
            temp_file.close()

            shutil.move(temp_file.name, deleted_log)
            if os.path.lexists(temp_file.name):
                os.remove(temp_file.name)

        except Exception as e:
            print(f"Error occurred while processing deleted files: {e}")
            temp_file.close()
            if os.path.lexists(temp_file.name):
                os.remove(temp_file.name)
            raise

    if base_commit_id:
        patch_info, patch_dir = prompt_info(main_path)
        generate_patch_file(main_path, base_commit_id, patch_info, patch_dir)

    return file_statuses


def prompt_info(main_path):
    # 1. task
    task = input("1. Select the task(train/inference/post_train): ").strip()
    while task not in ["train", "inference", "post_train"]:
        print("Invalid task. Must be one of: train, inference, post_train.")
        task = input("1. Select the task(train/inference/post_train): ").strip()

    # 2. backends
    third_party_dir = os.path.join(main_path, "third_party")
    available_submodules = [
        d for d in os.listdir(third_party_dir)
        if os.path.isdir(os.path.join(third_party_dir, d))
    ]

    while True:
        backend_input = input("2. Enter the backends (separate with commas): ").strip()
        backends = [b.strip() for b in backend_input.split(",") if b.strip()]
        invalid = [b for b in backends if b not in available_submodules]

        if not backends:
            print("At least one backend must be provided.")
        elif invalid:
            print(f"The following backends are not valid submodules: {', '.join(invalid)}")
            print(f"Available submodules: {', '.join(sorted(available_submodules))}")
        else:
            break

    backend_versions = {}
    for backend in backends:
        version = input(f"Enter the {backend} version: ").strip()
        backend_versions[backend] = version

    # 3. device type
    device_type = input("3. Enter the device type (e.g., NVIDIA_H100): ").strip()
    while device_type.count('_') != 1 or len(device_type.split('_')) != 2:
        print("Invalid format. Device type must be in the format xxx_yyy.")
        device_type = input("3. Enter the device type (e.g., NVIDIA_H100): ").strip()

    # === patch dir and yaml loading ===
    sorted_backends = sorted(backends)
    backend_key = "+".join(sorted_backends)
    patch_dir = os.path.join(main_path, "hardware", device_type, task, backend_key)

    yaml_fs_models = []
    yaml_contact = ""
    if os.path.exists(patch_dir):
        files = os.listdir(patch_dir)
        allowed = [f for f in files if f.endswith(".patch") or f.endswith(".yaml")]
        if len(files) != len(allowed) or len(allowed) > 2:
            raise RuntimeError(f"Directory {patch_dir} is not valid. Should only contain one .patch and one .yaml file.")
        for f in files:
            if f.endswith(".yaml"):
                try:
                    with open(os.path.join(patch_dir, f), "r") as f_yaml:
                        data = yaml.safe_load(f_yaml)
                        yaml_fs_models = data.get("fs_models", [])
                        yaml_contact = data.get("contact", "")
                except Exception as e:
                    print(f"Warning: Failed to load previous YAML config: {e}")

    # 4. FlagScale-compatible models (and show current models in patch dir)
    if yaml_fs_models:
        default_models = ",".join(yaml_fs_models)
        print(f"Current fs_models in patch directory: {', '.join(yaml_fs_models)}")
        model_input = input(f"4. Enter the FlagScale-compatible models (separated with commas) [{default_models}]: ").strip()
        model_input = model_input or default_models
    else:
        model_input = input("4. Enter the FlagScale-compatible models (separated with commas): ").strip()

    models = [m.strip() for m in model_input.split(",") if m.strip()]
    while not models:
        print("At least one FlagScale-compatible model must be provided.")
        model_input = input("4. Enter the FlagScale-compatible models (separated with commas): ").strip()
        models = [m.strip() for m in model_input.split(",") if m.strip()]

    # 5. Commit message
    commit_msg = input("5. Enter the commit message: ").strip()
    while not commit_msg:
        print("Commit message cannot be empty.")
        commit_msg = input("5. Enter the commit message: ").strip()

    # 6. Contact (optional)
    contact_prompt = "6. Contact info (optional): "
    contact_input = input(contact_prompt).strip()
    contact = contact_input or yaml_contact

    return {
        "task": task,
        "backends": sorted_backends,
        "backend_versions": backend_versions,
        "device_type": device_type,
        "fs_models": models,
        "contact": contact,
        "commit_msg": commit_msg
    }, patch_dir

def generate_patch_file(main_path: str, base_commit_id: str, patch_info: dict, patch_dir: str):
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
        
        # Step 1: stash all, including untracked, and create temp_branch_for_hardware_patch
        print("Step 1: Stashing current changes (including untracked) and create temp_branch_for_hardware_patch...")
        repo.git.stash("push", "--include-untracked")
        if 'temp_branch_for_hardware_patch' in repo.heads:
            print("Temporary branch 'temp_branch_for_hardware_patch' already exists, deleting...")
            repo.git.branch('-D', 'temp_branch_for_hardware_patch')
        repo.git.checkout("-b", 'temp_branch_for_hardware_patch')


        # Step2: apply stash.
        print("Step 2: Applying stashed changes to temp_branch_for_hardware_patch...")
        repo.git.stash("apply")
        repo.git.add(all=True)

        print("Step 3: Committing changes on temp_branch_for_hardware_patch...")
        # Step3: Commit with message "Patch for {base_commit_id}".
        commit_msg = patch_info["commit_msg"]
        repo.git.commit('-m', commit_msg)

        print("Step 4: Generating diff patch from base_commit_id to HEAD...")
        # Step4: Generate patch diff between base_commit_id and HEAD, writing into temp_file.
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
            tmp_patch_path = temp_file.name
            diff_data = repo.git.diff(base_commit_id, 'HEAD', '--binary')
            temp_file.write(diff_data)

        print("Step 5: Checking out base_commit_id...")
        # Step5: Checkout the base_commit_id commit.
        repo.git.checkout(base_commit_id)

        print("Step 6: Staging the generated patch file...")
        # Step6: Stage the  patch file.
        file_name = f"{base_commit_id[:7]}.patch"
        patch_file = os.path.join(patch_dir, file_name)
        patch_dir_exist = os.path.exists(patch_dir)
        os.makedirs(patch_dir, exist_ok=True)
        shutil.copy(tmp_patch_path, patch_file)
        repo.git.add(patch_file)
        if 'tmp_patch_path' in locals() and os.path.exists(tmp_patch_path):
            os.remove(tmp_patch_path)
            print(f"Temporary patch file {tmp_patch_path} deleted.")

        print("Step 7: Committing the patch file...")
        # Step7: Commit the patch file with the same message.
        repo.git.commit('-m', commit_msg)

        print("Step 8: Deleting temporary branch 'temp_branch_for_hardware_patch' locally...")
        # Step9: Delete the temporary branch locally.
        repo.git.branch('-D', 'temp_branch_for_hardware_patch')

        print(
            "Commit successfully! If you want to push,try 'git push origin HEAD:(your branch)' or  'git push --force origin HEAD:(your branch)'"
        )

    except Exception as e:
        print(f"Error: {e}")
        print(f"Rolling back to current branch...")
        repo.git.checkout(current_branch)
        repo.git.stash("pop")

    finally:
        try:
            if 'patch_dir_exist' in locals() and not patch_dir_exist:
                shutil.rmtree(patch_dir)
            if 'tmp_patch_path' in locals() and os.path.exists(tmp_patch_path):
                os.remove(tmp_patch_path)
                print(f"Temporary patch file {tmp_patch_path} deleted.")
            if 'temp_branch_for_hardware_patch' in repo.heads:
                repo.git.branch('-D', 'temp_branch_for_hardware_patch')
        except Exception as cleanup_error:
            print(f"Failed to delete temporary file: {cleanup_error}")
    # patch_dir_existed = os.path.exists(patch_dir)
    # file_name = f"{base_commit_id[:7]}.patch"
    # patch_file = os.path.join(patch_dir, file_name)
    # patch_file_created = os.path.exists(patch_file)

    # try:
    #     # Add all changes, including untracked files
    #     repo.git.add(all=True)

    #     # Ensure patch directory exists
    #     os.makedirs(patch_dir, exist_ok=True)

    #     # Check if patch file already exists
    #     if os.path.exists(patch_file):
    #         print(f"Patch file {patch_file} exists, updating...")
    #     else:
    #         print(f"Creating new patch file: {patch_file}")

    #     # Write diff to a temporary file first
    #     with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
    #         tmp_patch_path = temp_file.name
    #         diff_data = repo.git.diff(base_commit_id, cached=True)
    #         temp_file.write(diff_data)
        
    #     import time
    #     time.sleep(20)
    #     raise ValueError("CZ Test!!!")
    #     # Copy to final destination
    #     shutil.copy(tmp_patch_path, patch_file)
    #     print(f"Patch file generated/updated at: {patch_file}")

    # except Exception as e:
    #     print(f"Error generating patch: {e}")

    #     # Rollback Git changes: reset to original state
    #     try:
    #         repo.git.reset()
    #         print("Reset to original state.")
    #         # repo.index.reset(original_index)
    #         # for f in untracked_files:
    #         #     if os.path.exists(f):
    #         #         repo.git.rm(f, cached=True)
    #     except Exception as rollback_error:
    #         print(f"Rollback failed: {rollback_error}")
    # finally:
    #     # Cleanup temporary patch file
    #     try:
    #         if 'tmp_patch_path' in locals() and os.path.exists(tmp_patch_path):
    #             os.remove(tmp_patch_path)
    #             print(f"Temporary patch file {tmp_patch_path} deleted.")
    #     except Exception as cleanup_error:
    #         print(f"Failed to delete temporary file: {cleanup_error}")

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
                    print(
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
            print(f"The file {src_file_path} has been deleted.")
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
                print(
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
        print(f"The file {source_file} will be covered by {target_file}.")
    assert os.path.lexists(target_file)

    source_dir = os.path.dirname(source_file)
    if not os.path.lexists(source_dir):
        os.makedirs(source_dir, exist_ok=True)

    shutil.copyfile(target_file, source_file)
    if mode == "symlink":
        if os.path.lexists(target_file):
            os.remove(target_file)
        os.symlink(source_file, target_file)
        print(
            f"The file {target_file} has been copied to {source_file} and Create symbolic link {target_file} -> {source_file}."
        )
    elif mode == "copy":
        print(f"The file {source_file} has been copied to {target_file}.")
    else:
        raise ValueError(f"Unsupported mode: {mode}.")


def validate_base_commit(base_commit_id, main_path):
    main_repo = Repo(main_path)
    if base_commit_id:
        # Check if the base_commit_id exists in the FlagScale
        try:
            main_repo.commit(base_commit_id)
        except ValueError:
            raise ValueError(f"The commit ID {base_commit_id} does not exist in the FlagScale.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync submodule modifications to the corresponding backend in FlagScale."
    )
    parser.add_argument(
        "--backend",
        nargs="+",
        choices=["Megatron-LM", "vllm"],
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
        help="Base commit ID to reference in the repo. Default is None."
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

    for backend in backends:
        submodule_name = f"third_party/{backend}"
        src = None
        dst = os.path.join(main_path, "third_party", backend)
        # Megatron-LM
        if backend == "Megatron-LM":
            src = os.path.join(main_path, "flagscale", "train", "backends", backend)
            assert src
            patch(main_path, submodule_name, src, dst, mode, base_commit_id)
        # vllm
        if backend == "vllm":
            src = os.path.join(main_path, "flagscale", "inference", "backends", backend)
            assert src
            patch(main_path, submodule_name, src, dst, mode, base_commit_id)