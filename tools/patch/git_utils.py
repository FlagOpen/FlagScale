from logger_utils import get_patch_logger


logger = get_patch_logger()


def get_diff_between_commit_and_now(repo, commit):
    """
    Get the diff between the current state of the repository and a specific commit.

    Args:
        repo (git.Repo): The Git repository object.
        commit (str): The commit hash to compare against.

    Returns:
        diffs: A list of diffs between the current state and the specified commit, including:
            - Changes in the index (staged changes)
            - Changes in the working directory (unstaged changes)
            - Untracked files
    """
    commit_obj = repo.commit(commit)
    tree_obj = commit_obj.tree

    index_obj = repo.index
    index_tree_hash = index_obj.write_tree()

    staged_diff = tree_obj.diff(index_tree_hash)
    unstaged_diff = index_obj.diff(None)
    untracked_files = repo.untracked_files

    return staged_diff, unstaged_diff, untracked_files


def get_file_statuses_for_staged_or_unstaged(diffs):
    """
    Get the status of files.

    Args:
        diffs: The diff object containing changes.

    Returns:
        file_status: A dictionary with file paths as keys and their statuses as values.
    """
    file_statuses = {}
    for diff in diffs:
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

    logger.info(f"File statuses: {file_statuses}")
    return file_statuses


def get_file_statuses_for_untracked(untracked_files):
    """
    Get the status of untracked files.
    Args:
        untracked_files: A list of untracked files.
    Returns:
        file_status: A dictionary with file paths as keys and their statuses as values.
    """
    file_statuses = {}
    for file in untracked_files:
        file_statuses[file] = ["UT"]

    return file_statuses


from git import Repo, GitConfigParser
from pathlib import Path

def check_git_user_info(repo_path='.'):
    """
    Retrieve Git user.name and user.email from local config first,
    falling back to global config if not found. If neither is set,
    print a message suggesting how to configure them.

    Parameters:
        repo_path (str): Path to the Git repository (default: current directory).

    Returns:
        dict: Dictionary with 'name' and 'email' keys.
    """
    user_info = {'name': None, 'email': None}

    try:
        # Attempt to read from the local Git config
        repo = Repo(repo_path)
        local_config = repo.config_reader()

        if local_config.has_option('user', 'name'):
            user_info['name'] = local_config.get_value('user', 'name')
        if local_config.has_option('user', 'email'):
            user_info['email'] = local_config.get_value('user', 'email')
    except Exception as e:
        logger.warning(f"[local git config] Warning: {e}")

    try:
        # Fallback to global Git config if needed
        global_config = GitConfigParser([str(Path.home() / '.gitconfig')], read_only=True)

        if user_info['name'] is None and global_config.has_option('user', 'name'):
            user_info['name'] = global_config.get_value('user', 'name')
        if user_info['email'] is None and global_config.has_option('user', 'email'):
            user_info['email'] = global_config.get_value('user', 'email')
    except Exception as e:
        logger.warnin(f"[global git config] Warning: {e}")

    # Show warning if either field is still not set
    error_messages = ""
    if user_info['name'] is None:
        error_messages += "Git user.name is not set. You can set it with:\n  git config --global user.name YourName" + "\n"
    if user_info['email'] is None:
        error_messages += "Git user.email is not set. You can set it with:\n  git config --global user.email you@example.com"
    if error_messages:
        raise ValueError(error_messages)