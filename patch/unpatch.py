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
from exception import PathNotFound, GitApplyError, DirNotFound

path = os.getcwd()



def _add_auto_generate_args():
    parser = argparse.ArgumentParser(
        description="patch auto generate Arguments", allow_abbrev=False
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
    hetero_path = os.path.join(path, "patch/hetero.txt")
    if not os.path.exists(hetero_path):
        print("{} is not found!".format(hetero_path))
        raise FileNotFoundError
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
        print(patch_dir, " is not found!")
        raise FileNotFoundError
    files_and_folders = os.listdir(patch_dir)
    if len(files_and_folders) == 0:
        print(patch_dir, " have no file!")
        raise FileNotFoundError

    """get the base_commit_id stored in FlagScale """
    base_commit_id_now = [
        f for f in files_and_folders if os.path.isdir(os.path.join(patch_dir, f))
    ][0]
    """Check if the stored base_commit_id matches the input base_commit_id """

    if base_commit_id_now != base_commit_id:
        print("base_commit_id is not matched")
        raise FileNotFoundError
    base_commit_id_dir = os.path.join(patch_dir, base_commit_id)

    files_and_folders_1 = os.listdir(base_commit_id_dir)
    if len(files_and_folders_1) == 0:
        print(base_commit_id, " have no file!")
        raise FileNotFoundError
    patch_file = [
        f
        for f in files_and_folders_1
        if os.path.isfile(os.path.join(base_commit_id_dir, f))
    ][0]
    tmp_path = crete_tmp_dir(dir_path, tmp_str)
    file_name = save_unpatch_to_tmp(tmp_path, base_commit_id_dir, patch_file)
    repo.git.checkout(base_commit_id)
    try:
        repo.git.am(file_name)
    except:
        print("git apply {} falied!".format(file_name))
        raise GitApplyError
    shutil.rmtree(tmp_path)


def build_dir(repo, device_type, commit_id, directory=None):
    """Build directory for homogeneous scenarios."""
    global path
    if directory is None:
        apply_patch(repo, device_type, commit_id, path, "../tmp")
    else:
        if os.path.exists(os.path.join(path, directory)):
            shutil.rmtree(os.path.join(path, directory))
        dir_path = os.path.join(path, directory, device_type)
        build_dir_path = os.path.join(path, "../patch_build")
        if os.path.exists(build_dir_path):
            shutil.rmtree(build_dir_path)
        os.makedirs(build_dir_path)

        # copy FlagScale into build
        #os.system("cp -r {} {}".format(path, build_dir_path))
        shutil.copytree(path, build_dir_path)
        os.makedirs(dir_path)
        repo_name = path.split("/")[-1]
        # os.system("mv {} {}".format(os.path.join(build_dir_path, repo_name), dir_path))
        shutil.move(os.path.join(build_dir_path, repo_name), dir_path)
        # os.system(
        #     "mv {} {}".format(
        #         os.path.join(dir_path, repo_name), os.path.join(dir_path, "FlagScale")
        #     )
        # )
        shutil.move(os.path.join(dir_path, repo_name), os.path.join(dir_path, "FlagScale"))

        shutil.rmtree(build_dir_path)
        # step into build dir
        dir_path = os.path.join(dir_path, "FlagScale")
        repo = Repo(dir_path)
        apply_patch(repo, device_type, commit_id, dir_path, "../../../tmp")


def build_hetero_dir(repo, device_type, commit_id, directory):
    """Build directory for heterogeneous scenarios."""
    global path
    if os.path.exists(os.path.join(path, directory)):
        shutil.rmtree(os.path.join(path, directory))
    for device in device_type:
        dir_path = os.path.join(path, directory, device)
        build_dir_path = os.path.join(path, "../patch_build")
        if os.path.exists(build_dir_path):
            shutil.rmtree(build_dir_path)
        os.makedirs(build_dir_path)
        # step into build dir
        #os.system("cp -r {} {}".format(path, build_dir_path))
        shutil.copytree(path, build_dir_path)
        os.makedirs(dir_path)
        repo_name = path.split("/")[-1]
        # os.system("mv {} {}".format(os.path.join(build_dir_path, repo_name), dir_path))
        shutil.move(os.path.join(build_dir_path, repo_name), dir_path)
        # os.system(
        #     "mv {} {}".format(
        #         os.path.join(dir_path, repo_name), os.path.join(dir_path, "FlagScale")
        #     )
        # )
        shutil.move(os.path.join(dir_path, repo_name), os.path.join(dir_path, "FlagScale"))
        shutil.rmtree(build_dir_path)
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
        """Heterogeneous scenarios."""
        if args.dir is None:
            print("--dir must be set!")
            raise DirNotFound
        if check_hetero_txt(args.device_type, commit_id):
            build_hetero_dir(repo, args.device_type, commit_id, args.dir)
        else:
            print("The combination of device_type and commit_id is not in hetero.txt.")
            raise PathNotFound
    else:
        """Gomogeneous scenarios."""
        device_type = args.device_type[0]
        build_dir(repo, device_type, commit_id, args.dir)
    print("unpatch successfully!")


if __name__ == "__main__":
    main()
