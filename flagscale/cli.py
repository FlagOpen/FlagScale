import click
import os
import subprocess
import sys
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

def get_valid_backends_subsets(config_path="tests/scripts/unit_tests/config.yml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    VALID_BACKENDS_SUBSETS = {
        backend: list(subset_config["subset"].keys())
        for backend, subset_config in config.items()
    }
    
    return VALID_BACKENDS_SUBSETS

@flagscale.command()
@click.option(
    "--backend",
    "backend_name",
    required=True,
    type=str,
    help="The backend engine for testing",
)
@click.option(
    "--subset",
    "subset_name",
    required=True,
    type=str,
    help="Module name to be tested",
)
def unit_test(backend_name, subset_name):
    
    # Find the path of the flag_scale package
    flagscale_path = os.path.dirname(os.path.abspath(__file__))
    flag_scale_path = flagscale_path + "/../"
    os.chdir(flag_scale_path)

    VALID_BACKENDS_SUBSETS = get_valid_backends_subsets()
    
    # Validate the provided backend and subset
    if backend_name not in VALID_BACKENDS_SUBSETS:
        click.echo(f"Unsupported backend: {backend_name}. Supported backends: {', '.join(VALID_BACKENDS_SUBSETS.keys())}")
        return

    if subset_name not in VALID_BACKENDS_SUBSETS[backend_name]:
        # Generate supported combinations of backend and subset
        valid_combinations = [f"{backend} -> {subset}" for backend, subsets in VALID_BACKENDS_SUBSETS.items() for subset in subsets]
        click.echo(f"Unsupported subset: {subset_name} for backend: {backend_name}. Supported combinations: {', '.join(valid_combinations)}")
        return

    subprocess.run(["tests/scripts/unit_tests/test_subset.sh", "--backend", backend_name, "--subset", subset_name, "--coverage", "False"], check=True)


@flagscale.command()
def unit_test_all():
    # Find the path of the flag_scale package
    flagscale_path = os.path.dirname(os.path.abspath(__file__))
    flag_scale_path = flagscale_path + "/../"
    os.chdir(flag_scale_path)

    VALID_BACKENDS_SUBSETS = get_valid_backends_subsets()
    print(VALID_BACKENDS_SUBSETS)
    
    # Find the path of the flag_scale package
    flagscale_path = os.path.dirname(os.path.abspath(__file__))
    flag_scale_path = flagscale_path + "/../"
    os.chdir(flag_scale_path)
    for backend in VALID_BACKENDS_SUBSETS:
        for subset in VALID_BACKENDS_SUBSETS[backend]:
            subprocess.run(
                [
                    "tests/scripts/unit_tests/test_subset.sh",
                    "--backend", backend,
                    "--subset", subset,
                    "--coverage", "False"
                ],
                check=True
            )


if __name__ == "__main__":
    flagscale()
