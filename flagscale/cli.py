import os
import sys

import click

from run import main as run_main

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


if __name__ == "__main__":
    flagscale()
