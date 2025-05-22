import os
import shutil

from logger_utils import get_patch_logger, get_unpatch_logger


DELETED_FILE_NAME = "deleted_files.txt"


def sync_to_flagscale(file_path, status, src, dst, f=None, mode="symlink"):
    """
    Synchronize the file changes between the source and destination directories.
    Args:
        file_path (str): The path of the file relative to the root directory.
        status (list): A list containing the file status information.
        src (str): The source directory path, e.g., flagscale/backends/Megatron-LM.
        dst (str): The destination directory path, e.g., third_party/Megatron-LM.
        f (IOBase or None): An open file object to write the deleted files to.
        mode (str): The mode used to create the file. Default is "symlink".
                    if mode == "symlink":
                        Create a symbolic link from the source file to the destination file.
                    if mode == "copy":
                        Copy the source file to the destination file.
    """
    logger = get_patch_logger()

    src_file_path = os.path.join(src, file_path)
    dst_file_path = os.path.join(dst, file_path)
    change_type = status[0]

    SYMBOLIC_ERROR = "Defining symbolic links in the submodule is not supported except for those defined in FlagScale"
    TYPECHANGED_ERROR = (
        "File type changed are not supported in the submodule besides those defined by FlagScale"
    )

    # Process the file changes based on the change type.
    # Typechange
    if change_type == "T":
        is_symlink = os.path.islink(dst_file_path)
        if is_symlink:
            if not os.path.lexists(src_file_path):
                # The File is a symbolic link, but the source file no longer exists, so the symlink is dangling and has been automatically removed.
                logger.warning(
                    f"File {dst_file_path} is a symbolic link, but the source file no longer exists, so the symlink is dangling and has been automatically removed."
                )
                os.remove(dst_file_path)
        else:
            raise ValueError(f"{TYPECHANGED_ERROR}: {dst_file_path}")

    # New or Untracked
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
                    raise ValueError(f"{SYMBOLIC_ERROR}: {dst_file_path}")
        else:
            create_file(src_file_path, dst_file_path, mode=mode)

    # Deleted
    elif change_type == "D":
        if os.path.lexists(src_file_path):
            os.remove(src_file_path)
            logger.debug(f"File {src_file_path} has been deleted.")
        else:
            assert f
            f.write(f"{file_path}\n")
            f.flush()

    # Modified
    elif change_type == "M":
        is_symlink = os.path.islink(dst_file_path)
        if is_symlink:
            logger.warning(
                f"The symlink {dst_file_path} can only have a typechange status and it cannot have a modified status."
            )
        create_file(src_file_path, dst_file_path, mode=mode)

    # Renamed
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
            create_file(renamed_src_file_path, renamed_dst_file_path, mode=mode)
            assert f
            f.write(f"{file_path}\n")
            f.flush()


def create_file(source_file, target_file, mode="symlink"):
    """
    Create a file by copying or symlinking it from the source to the target.
    Args:
        source_file (str): Path of the file in the flagscale/backends/<backend>.
        target_file (str): Path of the file in the third_party/<submodule>.
        mode (str): Mode to use when creating the file. Can be either 'symlink' or 'copy'.
    """
    logger = get_patch_logger()

    if os.path.lexists(source_file):
        logger.warning(f"File {source_file} will be covered by {target_file}.")
    assert os.path.lexists(target_file)

    source_dir = os.path.dirname(source_file)
    if not os.path.lexists(source_dir):
        os.makedirs(source_dir, exist_ok=True)

    if os.path.isdir(target_file):
        logger.warning(f"File {target_file} is a directory.")
        for root, dirs, files in os.walk(target_file):
            for file in files:
                target_file_path = os.path.join(root, file)
                if ".git" in target_file_path:
                    logger.warning(f"Skip the .git* file {target_file_path}.")
                    continue
                relative_path = os.path.relpath(target_file_path, target_file)
                source_file_path = os.path.join(source_dir, relative_path)
                is_symlink = os.path.islink(target_file_path)
                if is_symlink:
                    if not os.path.lexists(source_file_path):
                        real_path = os.readlink(target_file_path)
                        if os.path.lexists(real_path):
                            os.makedirs(os.path.dirname(source_file_path), exist_ok=True)
                            shutil.move(real_path, source_file_path)
                            logger.info(
                                f"Move {real_path} to {source_file_path} and create symbolic link {target_file_path} -> {source_file_path}"
                            )
                            if os.path.lexists(target_file_path):
                                os.remove(target_file_path)

                else:
                    if not os.path.lexists(source_file_path):
                        os.makedirs(os.path.dirname(source_file_path), exist_ok=True)
                        shutil.copyfile(target_file_path, source_file_path)
                    else:
                        logger.warning(
                            f"File {source_file_path} will be covered by {target_file_path}."
                        )
                        shutil.copyfile(target_file_path, source_file_path)
                symlink_by_mode(source_file_path, target_file_path, mode=mode)
    else:
        shutil.copyfile(target_file, source_file)
        symlink_by_mode(source_file, target_file, mode=mode)


def symlink_by_mode(source_file, target_file, mode="symlink"):
    """
    Create a symbolic link from the source file to the target file.
    Args:
        source_file (str): Path of the file in the flagscale/backends/<backend>.
        target_file (str): Path of the file in the third_party/<submodule>.
        mode (str): Mode to use when creating the file. Can be either 'symlink' or 'copy'.
    """
    logger = get_patch_logger()

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


def copy(src, dst):
    logger = get_unpatch_logger()
    for root, dirs, files in os.walk(src):
        for filename in files:
            src_file = os.path.join(root, filename)
            if src_file == os.path.join(src, DELETED_FILE_NAME):
                continue

            rel_path = os.path.relpath(src_file, src)
            dst_file = os.path.join(dst, rel_path)

            dst_file_dir = os.path.dirname(dst_file)
            if not os.path.lexists(dst_file_dir):
                os.makedirs(dst_file_dir)

            if os.path.lexists(dst_file):
                os.remove(dst_file)
            assert not os.path.lexists(dst_file)
            shutil.copyfile(src_file, dst_file)
            logger.info(f"Copying file: {dst_file} -> {src_file}")


def delete_file(file_path, dst):
    logger = get_unpatch_logger()
    with open(file_path, "r", encoding="utf-8") as f:
        deleted_files = f.readlines()
        for deleted_file in deleted_files:
            deleted_file = deleted_file.strip()
            deleted_file_path = os.path.join(dst, deleted_file)
            if os.path.lexists(deleted_file_path):
                os.remove(deleted_file_path)
                logger.info(f"Deleting file: {deleted_file_path}")
            else:
                logger.warning(f"File not found for deletion: {deleted_file_path}")


def create_symlinks(src, dst):
    from unpatch import DELETED_FILE_NAME

    logger = get_unpatch_logger()
    for root, dirs, files in os.walk(src):
        for filename in files:
            src_file = os.path.join(root, filename)
            if src_file == os.path.join(src, DELETED_FILE_NAME):
                continue

            rel_path = os.path.relpath(src_file, src)
            dst_file = os.path.join(dst, rel_path)

            dst_file_dir = os.path.dirname(dst_file)
            if not os.path.lexists(dst_file_dir):
                os.makedirs(dst_file_dir)

            if os.path.lexists(dst_file):
                os.remove(dst_file)
            assert not os.path.lexists(dst_file)
            os.symlink(src_file, dst_file)

            logger.info(f"Creating symbolic link: {dst_file} -> {src_file}")
