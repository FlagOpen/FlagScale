import argparse
import os
import shutil
from git.repo import Repo
from common import (
    check_path,
    process_commit_id,
    git_init,
    crete_tmp_dir,
    save_unpatch_to_tmp,
)

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _add_auto_generate_args():
    parser = argparse.ArgumentParser(
        description="Patch auto generate Arguments", allow_abbrev=False
    )
    group = parser.add_argument_group(title="straggler")
    group.add_argument(
        "--device-type",
        type=str,
        nargs="+",
        required=True,
        help="Device type what you want to merge.",
    )
    group.add_argument(
        "--commit-id",
        type=str,
        required=True,
        help="The base commit-id that chip manufacturer must offer.",
    )
    group.add_argument(
        "--dir",
        type=str,
        default=None,
        help="The commit-id that want to patch.",
    )
    args = parser.parse_args()
    return args


def check_hetero_txt(device_type, base_commit_id):
    """Check if the combination of device_type and commit_id is in hetero.txt."""
    global path
    hetero_path = os.path.join(path, "tools/patch/hetero.txt")
    if not os.path.exists(hetero_path):
        raise FileNotFoundError("{} is not found!".format(hetero_path))
    with open(hetero_path, "r") as f:
        for line in f:
            line = line.strip()
            line = line.split(":")
            if base_commit_id == line[0].strip():
                hetero_list = line[1].split()
                if set(device_type).issubset(set(hetero_list)):
                    return True
    return False


def apply_patch(repo, device_type, base_commit_id, dir_path, tmp_str=None):
    """Convert FlagScale to in-place status by applying patch."""
    global path
    patch_dir = os.path.join(path, "hardwares", device_type)
    if not os.path.isdir(patch_dir):
        raise FileNotFoundError(patch_dir, " is not found!")
    files_and_folders = os.listdir(patch_dir)
    if len(files_and_folders) == 0:
        raise FileNotFoundError(patch_dir, " have no file!")

    # Get the base_commit_id stored in FlagScale.
    base_commit_id_now = [
        f for f in files_and_folders if os.path.isdir(os.path.join(patch_dir, f))
    ][0]

    # Check if the stored base_commit_id matches the input base_commit_id
    if base_commit_id_now != base_commit_id:
        raise FileNotFoundError("Base_commit_id is not matched")
    base_commit_id_dir = os.path.join(patch_dir, base_commit_id)

    files_and_folders_1 = os.listdir(base_commit_id_dir)
    if len(files_and_folders_1) == 0:
        raise FileNotFoundError(base_commit_id_dir, " have no file!")
    patch_file = [
        f
        for f in files_and_folders_1
        if os.path.isfile(os.path.join(base_commit_id_dir, f))
    ][0]
    tmp_path = crete_tmp_dir(dir_path, tmp_str)
    file_name = save_unpatch_to_tmp(tmp_path, base_commit_id_dir, patch_file)
    repo.git.checkout(base_commit_id)

    # Apply the patch file
    try:
        repo.git.am(file_name)
    except:
        raise ValueError("Git apply {} falied!".format(file_name))
    shutil.rmtree(tmp_path)


def build_dir(repo, device_type, commit_id, directory=None):
    """Build directory for homogeneous scenarios."""
    global path
    if directory is None:
        print("Now device is {} , processing....".format(device_type))
        apply_patch(repo, device_type, commit_id, path, "../tmp")
    else:
        print("Now device is {} , processing....".format(device_type))
        if os.path.exists(os.path.join(path, directory)):
            shutil.rmtree(os.path.join(path, directory))
        dir_path = os.path.join(path, directory, device_type)
        build_dir_path = os.path.join(path, "../patch_build")
        build_dir_path_dir = os.path.join(build_dir_path, directory)
        if os.path.exists(build_dir_path):
            shutil.rmtree(build_dir_path)

        # Copy FlagScale into build.
        shutil.copytree(path, build_dir_path)
        if os.path.isdir(build_dir_path_dir):
            shutil.rmtree(build_dir_path_dir)
        os.makedirs(dir_path)
        shutil.move(build_dir_path, dir_path)
        shutil.move(
            os.path.join(dir_path, "patch_build"), os.path.join(dir_path, "FlagScale")
        )

        # Step into build dir.
        dir_path = os.path.join(dir_path, "FlagScale")
        repo = Repo(dir_path)
        apply_patch(repo, device_type, commit_id, dir_path, "../../../tmp")


def build_hetero_dir(repo, device_type, commit_id, directory):
    """Build directory for heterogeneous scenarios."""
    global path
    if os.path.exists(os.path.join(path, directory)):
        shutil.rmtree(os.path.join(path, directory))
    for device in device_type:
        print("Now device is {} , processing....".format(device))
        dir_path = os.path.join(path, directory, device)
        build_dir_path = os.path.join(path, "../patch_build")
        build_dir_path_dir = os.path.join(build_dir_path, directory)
        if os.path.exists(build_dir_path):
            shutil.rmtree(build_dir_path)

        # Copy FlagScale into build.
        shutil.copytree(path, build_dir_path)
        if os.path.isdir(build_dir_path_dir):
            shutil.rmtree(build_dir_path_dir)
        os.makedirs(dir_path)
        shutil.move(build_dir_path, dir_path)
        shutil.move(
            os.path.join(dir_path, "patch_build"), os.path.join(dir_path, "FlagScale")
        )

        # Step into build dir.
        dir_path = os.path.join(dir_path, "FlagScale")
        repo = Repo(dir_path)
        apply_patch(repo, device, commit_id, dir_path, "../../../tmp")


def main():
    args = _add_auto_generate_args()
    check_path()
    global path
    repo = git_init(path)
    commit_id = process_commit_id(args.commit_id)
    if len(args.device_type) > 1:
        # Heterogeneous scenarios.
        if args.dir is None:
            raise FileNotFoundError("--dir must be set!")
        if check_hetero_txt(args.device_type, commit_id):
            build_hetero_dir(repo, args.device_type, commit_id, args.dir)
        else:
            raise NameError(
                "The combination of device_type and commit_id is not in hetero.txt."
            )
    else:
        # Homogeneous scenarios.
        device_type = args.device_type[0]
        build_dir(repo, device_type, commit_id, args.dir)
    print("Unpatch successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        if os.path.exists(os.path.join(path, "../tmp_flagscale/")):
            shutil.rmtree(os.path.join(path, "../tmp_flagscale/"))
