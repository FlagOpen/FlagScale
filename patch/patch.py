mport argparse
import os
import re
from git.repo import Repo

path = os.getcwd()


def check_path():
    """get path and check 'FlagScale' in path"""
    global path
    pattern = r".*FlagScale.*"
    a = re.match(pattern, path)
    if a is None:
        print("the FlagScale is not in your path")
        raise PathNotFound


def check_args(args):
    if args.device_type is None:
        print("args.device_type is None")
        raise PathNotFound
    if args.base_commit_id is None:
        print("args.base_commit_id is None")
        raise PathNotFound


def delete_dir(dir_path):
    os.system("rm -rf {}".format(dir_path))


class PathNotFound(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class GitNotFound(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class CommitShort(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class DeviceError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class UnpatchDiffError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def process_commit_id(base_commit_id, patch_commit_id):
    """Limit the length of the commit ID to 6"""
    if len(base_commit_id) >= 6:
        base_commit_id = base_commit_id[:6]
    else:
        print("base_commit_id is less longer than 6")
        raise CommitShort
    if len(patch_commit_id) >= 6:
        patch_commit_id = patch_commit_id[:6]
    else:
        print("patch_commit_id is less longer than 6")
        raise CommitShort
    return base_commit_id, patch_commit_id


def _add_auto_generate_args(parser):
    """set input argument"""
    group = parser.add_argument_group(title="straggler")
    group.add_argument(
        "--device-type",
        type=str,
        nargs="+",
        default=None,
        help="device type what you want to merge",
    )
    group.add_argument(
        "--base-commit-id",
        type=str,
        default=None,
        help="the base commit-id that chip manufacturer must offer",
    )
    group.add_argument(
        "--current-commit-id",
        type=str,
        default=None,
        help="the commit-id that want to patch",
    )
    return parser


def parse_autoargs():
    """parse the args of auto"""
    parser = argparse.ArgumentParser(
        description="patch auto generate Arguments", allow_abbrev=False
    )
    parser = _add_auto_generate_args(parser)
    args = parser.parse_args()
    return args


def get_device_type():
    """get the global variable device_type"""
    device_type = os.environ.get("DEVICE_TYPE", None)
    return device_type


def git_init(path=None):
    """git init the repo from path"""
    if not os.path.exists(path):
        cwd = os.getcwd()
        new_path = os.path.join(cwd, path)
        if not os.path.exists(new_path):
            raise PathNotFound
    try:
        repo = Repo(path)
    except:
        raise GitNotFound
    assert not repo.bare
    return repo


def get_current_commit(repo):
    """get the current commit id"""
    log = repo.git.log()
    commit_list = log.split("commit")
    current_commit = commit_list.split("\n")[0]
    return current_commit


def get_output_path(device_type, base_commit_id):
    global path
    device_path = os.path.join(path, "hardwares", str(device_type))
    patch_path = os.path.join(path, "hardwares", str(device_type), base_commit_id)
    if not os.path.isdir(device_path):
        os.makedirs(device_path)
    return device_path, patch_path


def check_hetero_txt(device_type, base_commit_id):
    """check if the combination of device_type and commit_id is in hetero.txt"""
    global path
    hetero_path = os.path.join(path, "patch/hetero.txt")
    with open(hetero_path, "r") as f:
        for line in f:
            line = line.strip()
            line = line.split(":")
            if base_commit_id == line[0].strip():
                hetero_list = line[1].split()
                if set(device_type).issubset(set(hetero_list)):
                    return True
        return False


def crete_tmp_dir():
    global path
    tmp_path = os.path.join(path, "../tmp")
    if not os.path.isdir(tmp_path):
        os.makedirs(tmp_path)
    return tmp_path


def save_patch_to_tmp(patch_name, patch_str):
    tmp_path = crete_tmp_dir()
    file_name = os.path.join(tmp_path, patch_name)
    with open(file_name, "w") as f:
        f.write(patch_str)


def get_patch(repo, device_type, base_commit_id, current_commit_id=None):
    """the main function to get the patch file"""
    if repo is None:
        print("repo is None")
        raise PathNotFound
    if current_commit_id is None:
        current_commit_id = repo.head.commit
    patch_str = repo.git.format_patch(
        "{}...{}".format(base_commit_id, current_commit_id), stdout=True
    )
    patch_name = "".join([base_commit_id, ".patch"])
    save_patch_to_tmp(patch_name, patch_str)
    repo.git.checkout(base_commit_id)
    device_path, patch_path = get_output_path(device_type, base_commit_id)
    update_patch(patch_str, patch_name, device_path, patch_path)
    auto_commit(repo, device_type, device_path, current_commit_id)


def get_hetero_patch(repo, device_type, base_commit_id, current_commit_id=None):

    global path
    if repo is None:
        print("repo is None")
        raise PathNotFound
    if current_commit_id is None:
        current_commit_id = repo.head.commit

    hetero_str = "{}: ".format(base_commit_id)
    for device in device_type[:-1]:
        hetero_str = hetero_str + " " + str(device)
        base_commit_id_path = os.path.join(path, "hardwares", device, base_commit_id)
        if not os.path.exists(base_commit_id_path):
            print("{} is not found".format(base_commit_id_path))
            raise PathNotFound
    now_device_type = device_type[-1]
    hetero_str = hetero_str + " " + str(now_device_type)
    patch_str = repo.git.format_patch(
        "{}...{}".format(base_commit_id, current_commit_id), stdout=True
    )
    patch_name = "".join([base_commit_id, ".patch"])
    save_patch_to_tmp(patch_name, patch_str)
    repo.git.checkout(base_commit_id)
    device_path, patch_path = get_output_path(now_device_type, base_commit_id)
    hetero_path = os.path.join(path, "patch/hetero.txt")
    update_patch(patch_str, patch_name, device_path, patch_path, hetero_path, hetero_str, device_type, base_commit_id)
    auto_commit(repo, now_device_type, device_path, current_commit_id)


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
        delete_dir(device_path)
    os.makedirs(patch_path)
    file_name = os.path.join(patch_path, patch_name)
    with open(file_name, "w") as f:
        f.write(patch_str)

    global path
    tmp_path = os.path.join(path, "../tmp")
    tmp_patch_path = os.path.join(tmp_path, file_name)
    os.system("cp {} {}".format(file_name, tmp_path))
    return tmp_path, tmp_patch_path


def auto_check(
    repo, base_commit_id, tmp_path, tmp_patch_path, origin_branch, unpatch_branch
):
    """check if origin code and unpatch code have different"""
    repo.git.checkout("-b", "unpatch_code")
    repo.git.checkout(base_commit_id)
    repo.git.am(tmp_patch_path)

    diff_str = repo.git.diff(origin_branch, unpatch_branch)
    if len(diff_str) > 0:
        print("WARNING: origin code and unpatch code have some different")
        repo.git.checkout("main")
        repo.git.branch("-D", "origin_patch_code")
        repo.git.branch("-D", "unpatch_code")
        raise UnpatchDiffError
    repo.git.checkout("main")
    repo.git.branch("-D", "origin_patch_code")
    repo.git.branch("-D", "unpatch_code")
    if os.path.isdir(tmp_path):
        delete_dir(tmp_path)


def auto_commit(repo, device_type, device_path, current_commit_id):
    """auto git commit the patch , commit-msg is from current_commit_id's commit-msg"""
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
    args = parse_autoargs()
    check_args(args)
    check_path()
    global path
    repo = git_init(path)
    base_commit_id, current_commit_id = process_commit_id(
        args.base_commit_id, args.current_commit_id
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

