import os
import subprocess
import sys

import click
import yaml

VERSION = "0.6.0"


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(
    VERSION, "-v", "--version", message=f"flagscale version {VERSION}"
)
def flagscale():
    """
    FlagScale is a comprehensive toolkit designed to support the entire lifecycle of large models,
    developed with the backing of the Beijing Academy of Artificial Intelligence (BAAI).
    """
    pass


@flagscale.command()
@click.argument("yaml_path", type=click.Path(exists=True))
def train(yaml_path):
    """
    Train model from yaml.
    """
    from run import main as run_main

    click.echo(f"Start training from the yaml {yaml_path}...")
    yaml_path = os.path.abspath(yaml_path)
    config_path = os.path.dirname(yaml_path)
    config_name = os.path.splitext(os.path.basename(yaml_path))[0]
    click.echo(f"config_path: {config_path}")
    click.echo(f"config_name: {config_name}")

    sys.argv = [
        "run.py",
        f"--config-path={config_path}",
        f"--config-name={config_name}",
    ]
    run_main()


@flagscale.command()
@click.argument("model_name", type=str)
@click.argument("yaml_path", type=click.Path(exists=True), required=False)
def serve(model_name, yaml_path=None):
    """
    Serve model from yaml.
    """
    from run import main as run_main

    if yaml_path:
        if os.path.isabs(yaml_path):
            yaml_path = yaml_path
        else:
            yaml_path = os.path.join(os.getcwd(), yaml_path)
        if not os.path.exists(yaml_path):
            click.echo(f"Error: The yaml {yaml_path} does not exist.", err=True)
            return
        click.echo(f"Using configuration yaml: {yaml_path}")
    else:
        default_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        yaml_path = os.path.join(
            default_dir, "examples", model_name, "conf", f"config_{model_name}.yaml"
        )
        if not os.path.exists(yaml_path):
            click.echo(f"Error: The yaml {yaml_path} does not exist.", err=True)
            return
        click.echo(f"Using default configuration yaml: {yaml_path}")
    click.secho(
        "Warning: When serving, please specify the relevant environment variables. When serving on multiple machines, ensure that the necessary parameters, such as hostfile, are set correctly. For details, refer to the following link: https://github.com/FlagOpen/FlagScale/blob/main/flagscale/serve/README.md",
        fg="yellow",
    )
    click.echo(f"Start serving from the yaml {yaml_path}...")
    yaml_path = os.path.abspath(yaml_path)
    config_path = os.path.dirname(yaml_path)
    config_name = os.path.splitext(os.path.basename(yaml_path))[0]
    click.echo(f"config_path: {config_path}")
    click.echo(f"config_name: {config_name}")

    sys.argv = [
        "run.py",
        f"--config-path={config_path}",
        f"--config-name={config_name}",
    ]
    run_main()


@flagscale.command()
# Input the name of the Docker image (required)
@click.option(
    "--image",
    "image_name",
    required=True,
    type=str,
    help="The name of the Docker image",
)
# Input the address of the Git repository (required)
@click.option(
    "--ckpt",
    "ckpt_name",
    required=True,
    type=str,
    help="The address of the ckpt's git repository",
)
# Input the address of the local directory (optional)
@click.option(
    "--ckpt-path",
    "ckpt_path",
    type=click.Path(),
    required=False,
    help="The path to save ckpt",
)
def pull(image_name, ckpt_name, ckpt_path):
    "Docker pull image and git clone ckpt."
    # If ckpt_path is not provided, use the default download directory
    if ckpt_path is None:
        ckpt_path = os.path.join(os.getcwd(), "model_download")

    # Check and create the directory
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
        print(f"Directory {ckpt_path} created.")

    # Pull the Docker image
    try:
        print(f"Pulling Docker image: {image_name}...")
        subprocess.run(["docker", "pull", image_name], check=True)
        print(f"Successfully pulled Docker image: {image_name}")
    except subprocess.CalledProcessError:
        print(f"Failed to pull Docker image: {image_name}")
        return

    # Clone the Git repository
    try:
        print(f"Cloning Git repository: {ckpt_name} into {ckpt_path}...")
        subprocess.run(["git", "clone", ckpt_name, ckpt_path], check=True)
        print(f"Successfully cloned Git repository: {ckpt_name}")
    except subprocess.CalledProcessError:
        print(f"Failed to clone Git repository: {ckpt_name}")
        return

    # Pull large files using Git LFS
    print("Pulling Git LFS files...")
    try:
        subprocess.run(["git", "lfs", "pull"], cwd=ckpt_path, check=True)
        print("Successfully pulled Git LFS files")
    except subprocess.CalledProcessError:
        print("Failed to pull Git LFS files")
        return


def change_to_flagscale():
    flagscale_path = os.path.dirname(os.path.abspath(__file__))
    flag_scale_path = flagscale_path + "/../"
    os.chdir(flag_scale_path)

    return flag_scale_path


def get_valid_backends_subsets(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    VALID_BACKENDS_SUBSETS = {}

    for backend, subset_config in config.items():
        subsets = list(subset_config["subset"].keys())
        VALID_BACKENDS_SUBSETS[backend] = subsets

    print(VALID_BACKENDS_SUBSETS)
    return VALID_BACKENDS_SUBSETS


def get_valid_types_tasks_cases(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    VALID_TYPES_TASKS_CASES = {}

    for test_type in config:
        VALID_TYPES_TASKS_CASES[test_type] = {}

        for task_name in config[test_type]:
            VALID_TYPES_TASKS_CASES[test_type][task_name] = []
            cases = config[test_type][task_name].strip().split()
            for case in cases:
                VALID_TYPES_TASKS_CASES[test_type][task_name].append(
                    case.lstrip("-").strip()
                )

    print(VALID_TYPES_TASKS_CASES)
    return VALID_TYPES_TASKS_CASES


def unit_test(backend_name, subset_name):
    change_to_flagscale()

    VALID_BACKENDS_SUBSETS = get_valid_backends_subsets(
        "tests/scripts/unit_tests/config.yml"
    )

    if backend_name not in VALID_BACKENDS_SUBSETS:
        click.echo(
            f"Unsupported backend: {backend_name}. Supported backends: {', '.join(VALID_BACKENDS_SUBSETS.keys())}"
        )
        return

    if subset_name not in VALID_BACKENDS_SUBSETS[backend_name]:
        valid_combinations = [
            f"{backend} -> {subset}"
            for backend, subsets in VALID_BACKENDS_SUBSETS.items()
            for subset in subsets
        ]
        click.echo(
            f"Unsupported subset: {subset_name} for backend: {backend_name}. Supported combinations: {', '.join(valid_combinations)}"
        )
        return

    subprocess.run(
        [
            "tests/scripts/unit_tests/test_subset.sh",
            "--backend",
            backend_name,
            "--subset",
            subset_name,
            "--coverage",
            "False",
        ],
        check=True,
    )


def unit_test_all():

    change_to_flagscale()

    VALID_BACKENDS_SUBSETS = get_valid_backends_subsets(
        "tests/scripts/unit_tests/config.yml"
    )

    for backend in VALID_BACKENDS_SUBSETS:
        for subset in VALID_BACKENDS_SUBSETS[backend]:
            subprocess.run(
                [
                    "tests/scripts/unit_tests/test_subset.sh",
                    "--backend",
                    backend,
                    "--subset",
                    subset,
                    "--coverage",
                    "False",
                ],
                check=True,
            )


def functional_test(type_name, task_name):

    flag_scale_path = change_to_flagscale()

    VALID_TYPES_TASKS_CASES = get_valid_types_tasks_cases(
        "tests/scripts/functional_tests/config.yml"
    )

    if type_name not in VALID_TYPES_TASKS_CASES:
        click.echo(
            f"Unsupported type: {type_name}. Supported types: {', '.join(VALID_TYPES_TASKS_CASES.keys())}"
        )
        return

    if task_name not in VALID_TYPES_TASKS_CASES[type_name]:
        valid_combinations = [
            f"{type_name} -> {task_name}"
            for type_name, tasks_name in VALID_TYPES_TASKS_CASES.items()
            for task_name in tasks_name
        ]
        click.echo(
            f"Unsupported task: {task_name} for type: {type_name}. Supported combinations: {', '.join(valid_combinations)}"
        )
        return

    try:
        subprocess.run(
            [
                "tests/scripts/functional_tests/test_task.sh",
                "--type",
                type_name,
                "--task",
                task_name,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: {e} \n")
        print("*" * 200)
        print(
            f"Also, please check the configuration file in path'{flag_scale_path}/tests/functional_tests/test_cases/{type_name}/{task_name}/conf' to ensure that the dependent files already exist."
        )
        print("*" * 200)
        raise


def functional_test_all():

    flag_scale_path = change_to_flagscale()

    VALID_TYPES_TASKS_CASES = get_valid_types_tasks_cases(
        "tests/scripts/functional_tests/config.yml"
    )

    for type_name in VALID_TYPES_TASKS_CASES:
        for task_name in VALID_TYPES_TASKS_CASES[type_name]:
            for case_name in VALID_TYPES_TASKS_CASES[type_name][task_name]:
                try:
                    subprocess.run(
                        [
                            "tests/scripts/functional_tests/test_task.sh",
                            "--type",
                            type_name,
                            "--task",
                            task_name,
                        ],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Error: {e} \n")
                    print("*" * 200)
                    print(
                        f"Also, please check the configuration file in path'{flag_scale_path}/tests/functional_tests/test_cases/{type_name}/{task_name}/conf' to ensure that the dependent files already exist."
                    )
                    print("*" * 200)
                    raise


@flagscale.command()
@click.option(
    "--unit",
    is_flag=True,
    help="Run specific unit test (requires --backend and --subset)",
)
@click.option(
    "--unit-all",
    is_flag=True,
    help="Run all unit tests (no additional parameters needed)",
)
@click.option(
    "--functional",
    is_flag=True,
    help="Run specific functional test (requires --type and --task)",
)
@click.option(
    "--functional-all",
    is_flag=True,
    help="Run all functional tests (no additional parameters needed)",
)
@click.option(
    "--backend", "backend_name", type=str, help="Backend name for unit testing"
)
@click.option("--subset", "subset_name", type=str, help="Subset name for unit testing")
@click.option(
    "--type", "type_name", type=str, help="Task classification for functional testing"
)
@click.option("--task", "task_name", type=str, help="Testing tasks that match the type")
def test(
    unit,
    unit_all,
    functional,
    functional_all,
    backend_name,
    subset_name,
    type_name,
    task_name,
):
    """Execute test command with flexible parameter requirements"""

    print(
        "unit, unit_all, functional, functional_all",
        unit,
        unit_all,
        functional,
        functional_all,
    )

    # Validate mutual exclusivity
    if unit and unit_all:
        raise click.UsageError("Cannot use both --unit and --unit-all")
    if functional and functional_all:
        raise click.UsageError("Cannot use both --functional and --functional-all")
    if (unit or unit_all) and (functional or functional_all):
        raise click.UsageError(
            "Cannot use both --unit/--unit-all with --functional/--functional-all"
        )

    # Unit test validation
    if unit:
        unit_test(backend_name, subset_name)
    elif unit_all:
        unit_test_all()

    # Functional test validation
    if functional:
        functional_test(type_name, task_name)
    elif functional_all:
        functional_test_all()

    if (not unit) and (not functional) and (not unit_all) and (not functional_all):
        unit_test_all()
        functional_test_all()


if __name__ == "__main__":
    flagscale()
