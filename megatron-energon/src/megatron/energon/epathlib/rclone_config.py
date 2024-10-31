import configparser
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ConfigEntry:
    name: str
    type: str
    provider: Optional[str]
    access_key_id: Optional[str]
    secret_access_key: Optional[str]
    region: Optional[str]
    endpoint: Optional[str]


def find_executable_path(executable_name):
    """Find the path of an executable in the PATH environment variable. Returns None if not found."""

    executable_path = shutil.which(executable_name)
    if executable_path:
        return Path(executable_path)
    return None


def get_rclone_config_path() -> Optional[Path]:

    # First check if rclone executable is in PATH, if yes, check if rclone.conf is in the same directory
    rclone_exe_path = find_executable_path("rclone")
    if rclone_exe_path is not None and rclone_exe_path.is_file():
        rclone_config_path = rclone_exe_path.with_name("rclone.conf")
        if rclone_config_path.is_file():
            return rclone_config_path

    # As a second option check the XDG_CONFIG_HOME environment variable, if it is set, check for rclone/rclone.conf in that directory
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home and Path(xdg_config_home).is_dir():
        rclone_config_path = Path(xdg_config_home) / "rclone" / "rclone.conf"
        if rclone_config_path.is_file():
            return rclone_config_path

    # As a third option check the default location ~/.config/rclone/rclone.conf
    rclone_config_path = Path.home() / ".config" / "rclone" / "rclone.conf"
    if rclone_config_path.is_file():
        return rclone_config_path

    # Last option is to check the legacy location ~/.rclone.conf
    legacy_config_path = Path.home() / ".rclone.conf"
    if legacy_config_path.is_file():
        return legacy_config_path

    return None


def read_rclone_config_at_path(config_path: Path) -> Dict[str, ConfigEntry]:
    """Reads the config file and returns a dictionary with the config entries."""

    config = configparser.ConfigParser()
    config.read(config_path)

    config_entries = {}
    for section in config.sections():
        entry = ConfigEntry(
            name=section,
            type=config[section].get("type"),
            provider=config[section].get("provider"),
            access_key_id=config[section].get("access_key_id"),
            secret_access_key=config[section].get("secret_access_key"),
            region=config[section].get("region"),
            endpoint=config[section].get("endpoint"),
        )
        config_entries[section] = entry

    return config_entries


def read_rclone_config() -> Dict[str, ConfigEntry]:
    config_path = get_rclone_config_path()
    if config_path is None:
        raise FileNotFoundError("Could not find rclone configuration file.")
    return read_rclone_config_at_path(config_path)
