import hydra
from omegaconf import DictConfig
from flagscale.logger import logger
from flagscale.launcher.runner import SSHRunner


@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig) -> None:
    runner = SSHRunner(config)

    if config.action == "run":
        runner.run()
    elif config.action == "dryrun":
        runner.run(dryrun=True)
    elif config.action == "test":
        runner.run(with_test=True)
    elif config.action == "stop":
        runner.stop()
    elif config.action == "query":
        runner.query()
    else:
        raise ValueError(f"Unknown action {config.action}")


if __name__ == "__main__":
    main()
