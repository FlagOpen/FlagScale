import argparse
import os
import shutil
from common import (
    check_path,
    process_commit_id,
    git_init,
    save_patch_to_tmp,
)
from exception import PathNotFound, DeviceError, UnpatchDiffError

path = os.getcwd()


def _add_auto_generate_args():
    """Set input argument."""
    parser = argparse.ArgumentParser(
        description="Patch auto generate Arguments.", allow_abbrev=False
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
        "--base-commit-id",
        type=str,
        required=True,
        help="The base commit-id that chip manufacturer must offer.",
    )
    group.add_argument(
        "--current-commit-id",
        type=str,
        default=None,
        help="The commit-id that want to patch.",
    )
    args = parser.parse_args()
    return args




def get_output_path(device_type, base_commit_id):
    global path
    device_path = os.path.join(path, "hardwares", str(device_type))
    patch_path = os.path.join(path, "hardwares", str(device_type), base_commit_id)
    if not os.path.isdir(device_path):
        os.makedirs(device_path)
    return device_path, patch_path


def check_hetero_txt(device_type, base_commit_id):
    """Check if the combination of device_type and commit_id is in hetero.txt."""
    global path
    hetero_path = os.path.join(path, "patch/hetero.txt")
    if not os.path.exists(hetero_path):
        os.system("touch {}".format(hetero_path))
    with open(hetero_path, "r") as f:
        for line in f:
            line = line.strip()
            line = line.split(":")
            if base_commit_id == line[0].strip():
                hetero_list = line[1].split()
                if set(device_type).issubset(set(hetero_list)):
                    return True
    return False


def get_patch(repo, device_type, base_commit_id, current_commit_id=None):
    """The main function to get the patch file."""
    if repo is None:
        print("repo is None")
        raise PathNotFound
    global path
    
    # Create diretory to save patch.py/unpatch.py.
    patch_file_path = os.path.join(path, "patch/")
    tmp_patch_file_path = os.path.join(path, "../tmp_patch/")
    if os.path.exists(tmp_patch_file_path):
        shutil.rmtree(tmp_patch_file_path)
    shutil.copytree(patch_file_path,tmp_patch_file_path)
    
    # Create in-place code branch to compare different.
    origin_patch_branch = "origin_patch_code"
    try:
        repo.git.stash()
        repo.git.checkout(current_commit_id)
        repo.git.branch("origin_patch_code")
    except:
        print("branch {} is exist!".format(origin_patch_branch))
        raise FileExistsError
    patch_str = repo.git.format_patch(
        "{}...{}".format(base_commit_id, current_commit_id), stdout=True
    )
    patch_name = "".join([base_commit_id, ".patch"])
    file_name, tmp_path = save_patch_to_tmp(patch_name, patch_str)
    repo.git.stash()
    repo.git.checkout(base_commit_id)
    
    # Create patch code branch to compare different.
    try:
        unpatch_branch = "unpatch_code"
        repo.git.branch("unpatch_code")
    except:
        print("branch {} is exist!".format(unpatch_branch))
        raise FileExistsError
    # Check the different between in-place code 
    auto_check(repo, file_name, base_commit_id, origin_patch_branch, unpatch_branch)
    shutil.rmtree(tmp_path)
    device_path, patch_path = get_output_path(device_type, base_commit_id)
    if not os.path.exists(patch_file_path):
        shutil.copytree(tmp_patch_file_path,os.path.join(path,'patch'))
    else:
        shutil.rmtree(os.path.join(path,'patch'))
        shutil.copytree(tmp_patch_file_path, os.path.join(path,'patch'))
    shutil.rmtree(tmp_patch_file_path)
    update_patch(patch_str, patch_name, device_path, patch_path)
    auto_commit(repo, device_type, device_path, current_commit_id)


def get_hetero_patch(repo, device_type, base_commit_id, current_commit_id=None):
    global path
    if repo is None:
        print("repo is None")
        raise PathNotFound
    hetero_str = "{}: ".format(base_commit_id)
    for device in device_type[:-1]:
        hetero_str = hetero_str + " " + str(device)
        base_commit_id_path = os.path.join(path, "hardwares", device, base_commit_id)
        if not os.path.exists(base_commit_id_path):
            print("{} is not found".format(base_commit_id_path))
            raise PathNotFound
    now_device_type = device_type[-1]
    hetero_str = hetero_str + " " + str(now_device_type)
    patch_file_path = os.path.join(path, "patch/")
    tmp_patch_file_path = os.path.join(path, "../tmp_patch/")
    if os.path.exists(tmp_patch_file_path):
        shutil.rmtree(tmp_patch_file_path)
    shutil.copytree(patch_file_path,tmp_patch_file_path)
    try:
        repo.git.stash()
        repo.git.checkout(current_commit_id)
        origin_patch_branch = "origin_patch_code"
        repo.git.branch("origin_patch_code")
    except:
        print("branch {} is exist!".format(origin_patch_branch))
        raise FileExistsError
    patch_str = repo.git.format_patch(
        "{}...{}".format(base_commit_id, current_commit_id), stdout=True
    )
    patch_name = "".join([base_commit_id, ".patch"])
    file_name, tmp_path = save_patch_to_tmp(patch_name, patch_str)
    patch_file_path = os.path.join(path, "patch/")
    repo.git.stash()

    repo.git.checkout(base_commit_id)
    try:
        unpatch_branch = "unpatch_code"
        repo.git.branch("unpatch_code")
    except:
        print("branch {} is exist!".format(unpatch_branch))
        raise FileExistsError
    auto_check(repo, file_name, base_commit_id, origin_patch_branch, unpatch_branch)
    shutil.rmtree(tmp_path)
    device_path, patch_path = get_output_path(now_device_type, base_commit_id)
    if not os.path.exists(patch_file_path):
        shutil.copytree(tmp_patch_file_path,os.path.join(path,'patch'))
    else:
        shutil.rmtree(os.path.join(path,'patch'))
        shutil.copytree(tmp_patch_file_path, os.path.join(path,'patch'))
    shutil.rmtree(tmp_patch_file_path)
    hetero_path = os.path.join(path, "patch/hetero.txt")
    update_patch(
        patch_str,
        patch_name,
        device_path,
        patch_path,
        hetero_path,
        hetero_str,
        device_type,
        base_commit_id,
    )
    auto_commit(repo, now_device_type, device_path, current_commit_id, hetero_path)


def update_patch(
    patch_str,
    patch_name,
    device_path,
    patch_path,
    hetero_path=None,
    hetero_str=None,
    device_type=None,
    base_commit_id=None,
):
    """hetero_path is not None then hetero_str must be not None"""
    assert bool(hetero_path) == bool(hetero_str)
    """write to hetero.txt"""
    if hetero_str:
        if not check_hetero_txt(device_type, base_commit_id):
            with open(hetero_path, "a+") as f:
                f.writelines(hetero_str + "\n")
    if os.path.isdir(device_path):
        shutil.rmtree(device_path)
    os.makedirs(patch_path)
    file_name = os.path.join(patch_path, patch_name)
    with open(file_name, "w") as f:
        f.write(patch_str)


def auto_check(repo, file_name, base_commit_id, origin_branch, unpatch_branch):
    """check if origin code and unpatch code have different"""
    repo.git.checkout(unpatch_branch)
    repo.git.am(file_name)
    diff_str = repo.git.diff(origin_branch, unpatch_branch)
    if len(diff_str) > 0:
        print("WARNING: origin code and unpatch code have some different")
        repo.git.stash()
        repo.git.checkout("main")
        repo.git.checkout(base_commit_id)
        repo.git.branch("-D", "origin_patch_code")
        repo.git.branch("-D", "unpatch_code")
        raise UnpatchDiffError
    print("auto check successfully!")
    repo.git.stash()
    try:
        repo.git.checkout("main")
    except:
        import traceback

        traceback.print_exc()
    repo.git.checkout(base_commit_id)
    repo.git.branch("-D", "origin_patch_code")
    repo.git.branch("-D", "unpatch_code")


def auto_commit(repo, device_type, device_path, current_commit_id, hetero_path=None):
    """auto git commit the patch , commit-msg is from current_commit_id's commit-msg"""
    if hetero_path:
        repo.git.add(hetero_path)
    repo.git.add(device_path)
    commit_msg = repo.git.log(current_commit_id)
    commit_msg = commit_msg.split("\ncommit")[0].split("\n")[4].strip()
    if len(commit_msg.split("]")) == 1:
        commit_msg = commit_msg
    else:
        commit_msg = commit_msg.split("]")[1].strip()
    commit_msg = "[{}] {}".format(device_type, commit_msg)
    repo.git.commit("-m", commit_msg)


def check_device_type(device_type):
    """Check the format of device_type. The device_type format must be '--device-type A_X100' or  '--device-type A_X100  B_Y100'
    '--device-type A_X100'  format for homogeneous scenarios
    '--device-type A_X100  B_Y100' format for heterogeneous scenarios
    """
    import re

    device_pattern = r"\w+"
    for device in device_type:
        match = re.fullmatch(device_pattern, device)
        if not match:
            return False
    return True


def main():
    args = _add_auto_generate_args()
    check_path()
    global path
    repo = git_init(path)
    if args.current_commit_id is None:
        current_commit_id = repo.head.commit
    else:
        current_commit_id = args.current_commit_id
    current_commit_id, base_commit_id = process_commit_id(
        current_commit_id, args.base_commit_id
    )
    if not check_device_type(args.device_type):
        print("device_type is not legal!")
        raise DeviceError

    if len(args.device_type) > 1:
        """heterogeneous scenarios"""
        get_hetero_patch(repo, args.device_type, base_commit_id, current_commit_id)
    else:
        """homogeneous scenarios"""
        get_patch(repo, args.device_type[0], base_commit_id, current_commit_id)
    print("patch successfully!")


if __name__ == "__main__":
    main()
