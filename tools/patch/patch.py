import argparse
import os
import shutil
import sys
import tempfile

from git.repo import Repo


def patch(main_path, submodule_name, src, dst, mode="symlink"):
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
    tmp_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    for file_path in file_statuses:
        if file_statuses[file_path][0] == "D":
            file_status_deleted[file_path] = file_statuses[file_path]

    for file_path in file_statuses:
        if file_statuses[file_path][0] == "D":
            continue
        _sync(file_path, file_statuses[file_path], src, dst, tmp_file, mode=mode)

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
                    tmp_file,
                    mode=mode,
                )
            deleted_log = os.path.join(src, "deleted_files.log")
            tmp_file.close()

            shutil.move(tmp_file.name, deleted_log)
            if os.path.lexists(tmp_file.name):
                os.remove(tmp_file.name)

        except Exception as e:
            print(f"Error occurred while processing deleted files: {e}")
            tmp_file.close()
            if os.path.lexists(tmp_file.name):
                os.remove(tmp_file.name)
            raise

    return file_statuses


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
        if not os.path.lexists(src_file_path):
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
        help="Mode to patch (default: symlink)",
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
            patch(main_path, submodule_name, src, dst, args.mode)
        # vllm
        if backend == "vllm":
            src = os.path.join(main_path, "flagscale", "inference", "backends", backend)
            assert src
            patch(main_path, submodule_name, src, dst, args.mode)
