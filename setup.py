import os
import subprocess

from setuptools import find_packages, setup
from setuptools.command.install import install


def is_nvidia_chip():
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode == 0:
            return True
    except Exception as e:
        pass
    return False


class InstallRequirement(install):
    def run(self):
        if is_nvidia_chip():
            # TODO: At present, it is assumed that the relevant dependencies have been installed,
            # and automatic installation of the relevant dependencies is supported in the future.
            # After verification, open the comment.
            """
            print("In NVIDIA chip, install train and inference dependencies automatically...")
            # Install train requirements
            print("Install train dependencies...")
            env_param = '--env train'
            subprocess.check_call(['bash', './install/install-requirements.sh', env_param])

            # Install vllm requirements
            print("Install inference dependencies...")
            env_param = "--env inference"
            subprocess.check_call(
                ["bash", "./install/install-requirements.sh", env_param]
            )
            """
            pass
        # Continue to install
        install.run(self)


setup(
    name="flag_scale",
    version="0.6.0",
    description="FlagScale is a comprehensive toolkit designed to support the entire lifecycle of large models, developed with the backing of the Beijing Academy of Artificial Intelligence (BAAI). ",
    url="https://github.com/FlagOpen/FlagScale",
    packages=[
        "flag_scale",
        "flag_scale.megatron",
        "flag_scale.flagscale",
        "flag_scale.examples",
        "flag_scale.communication",
    ],
    package_dir={
        "flag_scale": "",
        "flag_scale.megatron": "megatron",
        "flag_scale.flagscale": "flagscale",
        "flag_scale.examples": "examples",
        "flag_scale.communication": "communication",
    },
    package_data={
        "flag_scale.megatron": ["**/*"],
        "flag_scale.flagscale": ["**/*"],
        "flag_scale.examples": ["**/*"],
        "flag_scale.communication": ["**/*"],
    },
    install_requires=[
        "click",
        "cryptography",
        "setuptools>=75.1.0",
        "packaging>=24.1",
        "importlib_metadata>=8.5.0"
    ],

    entry_points={
        "console_scripts": [
            "flagscale=flag_scale.flagscale.cli:flagscale",
        ],
    },
    cmdclass={"install": InstallRequirement},
)
