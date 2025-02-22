import argparse
import os
import shutil

from common import (
    check_branch_name,
    check_path,
    decrypt_file,
    encrypt_file,
    get_now_branch_name,
    git_init,
    process_commit_id,
    save_patch_to_tmp,
)

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    group.add_argument(
        "--key-path",
        type=str,
        default=None,
        help="The path for storing public and private keys. Be careful not to upload to the Git repository.",
    )
    args = parser.parse_args()
    return args


def get_output_path(device_type, base_commit_id):
    """Get the output path to save patch file in hardware directory."""
    global path
    device_path = os.path.join(path, "hardware", str(device_type))
    patch_path = os.path.join(path, "hardware", str(device_type), base_commit_id)
    if not os.path.isdir(device_path):
        os.makedirs(device_path)
    return device_path, patch_path


def check_hetero_txt(device_type, base_commit_id):
    """Check if the combination of device_type and commit_id is in hetero.txt."""
    global path
    hetero_path = os.path.join(path, "tools/hetero.txt")
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


def get_patch(repo, device_type, base_commit_id, current_commit_id=None, key_path=None):
    """The main function to get the patch file in homogeneous scenarios."""
    if repo is None:
        raise FileNotFoundError("Repo is None")
    global path

    # Create diretory to save patch.py/unpatch.py.
    patch_file_path = os.path.join(path, "tools/patch")
    tmp_patch_file_path = os.path.join(path, "../tmp_patch")
    if os.path.exists(tmp_patch_file_path):
        shutil.rmtree(tmp_patch_file_path)
    shutil.copytree(patch_file_path, tmp_patch_file_path)

    # Create in-place code branch to compare different.
    origin_patch_branch = "origin_patch_code"
    now_branch = get_now_branch_name(repo)
    repo.git.stash()
    repo.git.checkout(current_commit_id)

    if check_branch_name(repo, origin_patch_branch):
        repo.git.branch("-D", origin_patch_branch)
        repo.git.branch(origin_patch_branch)
    else:
        repo.git.branch(origin_patch_branch)

    patch_str = repo.git.format_patch(
        "{}...{}".format(base_commit_id, current_commit_id), stdout=True
    )

    # Save .patch file to tmp directory.
    patch_name = "".join([base_commit_id, ".patch"])
    file_name, tmp_path = save_patch_to_tmp(patch_name, patch_str, key_path)
    repo.git.stash()
    repo.git.checkout(base_commit_id)

    # Create patch code branch to compare different.
    unpatch_branch = "unpatch_code"
    if check_branch_name(repo, unpatch_branch):
        repo.git.branch("-D", unpatch_branch)
        repo.git.branch(unpatch_branch)
    else:
        repo.git.branch(unpatch_branch)

    # Check the different between in-place code and patch code.
    auto_check(
        repo,
        file_name,
        base_commit_id,
        now_branch,
        origin_patch_branch,
        unpatch_branch,
        key_path,
    )
    shutil.rmtree(tmp_path)
    device_path, patch_path = get_output_path(device_type, base_commit_id)
    print(device_path, patch_path)

    # Recover the patch/ directory.
    if not os.path.exists(patch_file_path):
        shutil.copytree(tmp_patch_file_path, os.path.join(path, "tools/patch"))
    else:
        shutil.rmtree(os.path.join(path, "tools/patch"))
        shutil.copytree(tmp_patch_file_path, os.path.join(path, "tools/patch"))
    shutil.rmtree(tmp_patch_file_path)
    update_patch(patch_str, patch_name, device_path, patch_path, key_path=key_path)
    auto_commit(repo, device_type, device_path, current_commit_id)


def get_hetero_patch(
    repo, device_type, base_commit_id, current_commit_id=None, key_path=None
):
    """The main function to get the patch file in heterogeneous scenarios."""
    global path
    if repo is None:
        TypeError("repo is None")
    hetero_str = "{}: ".format(base_commit_id)
    for device in device_type[:-1]:
        hetero_str = hetero_str + " " + str(device)
        base_commit_id_path = os.path.join(path, "hardware", device, base_commit_id)
        if not os.path.exists(base_commit_id_path):
            raise FileNotFoundError("{} is not found".format(base_commit_id_path))
    now_device_type = device_type[-1]
    hetero_str = hetero_str + " " + str(now_device_type)

    # Create diretory to save patch.py/unpatch.py.
    patch_file_path = os.path.join(path, "tools/patch")
    tmp_patch_file_path = os.path.join(path, "../tmp_patch")
    if os.path.exists(tmp_patch_file_path):
        shutil.rmtree(tmp_patch_file_path)
    shutil.copytree(patch_file_path, tmp_patch_file_path)
    now_branch = get_now_branch_name(repo)
    repo.git.stash()
    repo.git.checkout(current_commit_id)

    # Create in-place code branch to compare different.

    origin_patch_branch = "origin_patch_code"
    if check_branch_name(repo, origin_patch_branch):
        repo.git.branch("-D", origin_patch_branch)
        repo.git.branch(origin_patch_branch)
    else:
        repo.git.branch(origin_patch_branch)

    patch_str = repo.git.format_patch(
        "{}...{}".format(base_commit_id, current_commit_id), stdout=True
    )
    patch_name = "".join([base_commit_id, ".patch"])

    # Create in-place code branch to compare different.
    file_name, tmp_path = save_patch_to_tmp(patch_name, patch_str, key_path=None)
    patch_file_path = os.path.join(path, "tools/patch")
    repo.git.stash()

    repo.git.checkout(base_commit_id)

    unpatch_branch = "unpatch_code"
    if check_branch_name(repo, unpatch_branch):
        repo.git.branch("-D", unpatch_branch)
        repo.git.branch(unpatch_branch)
    else:
        repo.git.branch(unpatch_branch)

    # Check the different between in-place code and patch code.
    auto_check(
        repo, file_name, base_commit_id, now_branch, origin_patch_branch, unpatch_branch
    )
    shutil.rmtree(tmp_path)
    device_path, patch_path = get_output_path(now_device_type, base_commit_id)

    # Recover the patch/ directory.
    if not os.path.exists(patch_file_path):
        shutil.copytree(tmp_patch_file_path, os.path.join(path, "tools/patch"))
    shutil.rmtree(tmp_patch_file_path)
    hetero_path = os.path.join(path, "tools/patch/hetero.txt")

    # Update .patch file and hetero.txt.
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
    key_path=None,
):
    """Hetero_path is not None then hetero_str must be not None."""
    assert bool(hetero_path) == bool(hetero_str)
    # Write to hetero.txt.
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
    if key_path is not None:
        public_key_path = os.path.join(key_path, "public_key.pem")
        encrypt_file(file_name, public_key_path)  # Encrypt the specified file
        # Delete the original patch file
        os.remove(file_name)


def auto_check(
    repo,
    file_name,
    base_commit_id,
    now_branch,
    origin_branch,
    unpatch_branch,
    key_path=None,
):
    """Check if origin code and unpatch code have different."""
    repo.git.checkout(unpatch_branch)
    if key_path is not None:
        private_key_path = os.path.join(key_path, "private_key.pem")
        decrypt_file(file_name, private_key_path)  # Decrypt the file
        file_name = os.path.splitext(file_name)[0]  # This will remove the extension
    repo.git.am(file_name)
    diff_str = repo.git.diff(origin_branch, unpatch_branch)
    if len(diff_str) > 0:
        print("WARNING: origin code and unpatch code have some different")
        repo.git.stash()
        repo.git.checkout(now_branch)
        repo.git.checkout(base_commit_id)
        repo.git.branch("-D", "origin_patch_code")
        repo.git.branch("-D", "unpatch_code")
        raise ValueError
    print("Auto check successfully!")
    repo.git.stash()
    repo.git.checkout(now_branch)
    repo.git.checkout(base_commit_id)
    repo.git.branch("-D", "origin_patch_code")
    repo.git.branch("-D", "unpatch_code")


def auto_commit(repo, device_type, device_path, current_commit_id, hetero_path=None):
    """Auto git commit the patch , commit-msg is from current_commit_id's commit-msg."""
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
    print(
        "Commit successfully! If you want to push,try 'git push origin HEAD:(your branch)' or  'git push --force origin HEAD:(your branch)'"
    )


def check_device_type(device_type):
    """Check the format of device_type. The device_type format must be '--device-type A_X100' or  '--device-type A_X100  B_Y100'
    '--device-type A_X100'  format for homogeneous scenarios.
    '--device-type A_X100  B_Y100' format for heterogeneous scenarios.
    """
    import re

    device_pattern = r"\w+_\w+"
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
        current_commit_id = str(repo.head.commit)
    else:
        current_commit_id = args.current_commit_id
    current_commit_id, base_commit_id = process_commit_id(
        current_commit_id, args.base_commit_id
    )
    if not check_device_type(args.device_type):
        raise SyntaxError("device_type is not legal!")

    if len(args.device_type) > 1:
        # Heterogeneous scenarios.
        get_hetero_patch(repo, args.device_type, base_commit_id, current_commit_id)
    else:
        # Homogeneous scenarios.
        get_patch(
            repo, args.device_type[0], base_commit_id, current_commit_id, args.key_path
        )
    print("Patch successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        path = os.getcwd()
        if os.path.exists(os.path.join(path, "../tmp_patch/")):
            shutil.rmtree(os.path.join(path, "../tmp_patch/"))
        if os.path.exists(os.path.join(path, "../tmp_flagscale/")):
            shutil.rmtree(os.path.join(path, "../tmp_flagscale/"))
