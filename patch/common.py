import re
import os
import shutil
from git.repo import Repo

path = os.getcwd()


def check_path():
    """Get path and check 'FlagScale' in path."""
    global path
    pattern = r".*FlagScale.*"
    a = re.match(pattern, path)
    if a is None:
        raise FileNotFoundError("the FlagScale is not in your path")


def process_commit_id(patch_commit_id, base_commit_id=None):
    """Limit the length of the commit ID to 6."""
    if base_commit_id is not None:
        if len(base_commit_id) >= 6:
            base_commit_id = base_commit_id[:6]
        else:
            raise ValueError("base_commit_id is less longer than 6")
    if len(patch_commit_id) >= 6:
        patch_commit_id = patch_commit_id[:6]
    else:
        raise ValueError("patch_commit_id is less longer than 6")
    if base_commit_id is not None:
        return patch_commit_id, base_commit_id
    else:
        return patch_commit_id


def git_init(path=None):
    """Git init the repo from path."""
    if path:
        if not os.path.exists(path):
            cwd = os.getcwd()
            new_path = os.path.join(cwd, path)
            if not os.path.exists(new_path):
                raise FileNotFoundError(new_path)
    check_path()
    try:
        repo = Repo(path)
    except:
        raise FileNotFoundError(path)
    assert not repo.bare
    return repo


def crete_tmp_dir(dir_path=None, tmp_str=None):
    global path
    if dir_path is None:
        tmp_path = os.path.join(path, "../tmp_flagscale")
    else:
        if tmp_str is not None:
            tmp_path = os.path.join(dir_path, tmp_str.replace("tmp", "tmp_flagscale"))
        else:
            tmp_path = os.path.join(dir_path, "../tmp_flagscale")
    if not os.path.isdir(tmp_path):
        os.makedirs(tmp_path)
    else:
        shutil.rmtree(tmp_path)
        os.makedirs(tmp_path)
    return tmp_path


def check_branch_name(repo, branch_name):
    """Check if branch_name exists in the repository."""
    branch_list = repo.git.branch("--list")
    if branch_name in branch_list:
        return True
    else:
        return False


def save_patch_to_tmp(patch_name, patch_str):
    """Save patch str to tmp patch file."""
    tmp_path = crete_tmp_dir()
    file_name = os.path.join(tmp_path, patch_name)
    with open(file_name, "w") as f:
        f.write(patch_str)
    return file_name, tmp_path


def save_unpatch_to_tmp(tmp_path, base_commit_id_dir, patch_file):
    """Save patch file to tmp directory."""
    file_name = os.path.join(base_commit_id_dir, patch_file)
    try:
        shutil.copy(file_name, tmp_path)
    except:
        raise ValueError("{} cannot cp".format(file_name))
    tmp_file_name = os.path.join(tmp_path, patch_file)
    return tmp_file_name
