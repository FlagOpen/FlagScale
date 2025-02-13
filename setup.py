from setuptools import setup, find_packages


setup(
    name="flagscale",
    version="0.6.0",
    description="A CLI tool for FlagScale",
    url="https://github.com/FlagOpen/FlagScale",
    packages=find_packages(),
    install_requires=["click"],
    entry_points={
        "console_scripts": [
            "flagscale=flagscale.cli:flagscale",
        ],
    },
)
