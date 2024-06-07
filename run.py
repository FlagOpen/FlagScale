import hydra
from omegaconf import DictConfig
from flagscale.logger import logger
from flagscale.launcher.runner import SSHRunner
from flagscale.launcher.runner import CloudRunner


@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig) -> None:
    if config.action == "auto_tune":
        from flagscale.auto_tuner import AutoTuner
        # For MPIRUN scene, just one autotuner process.
        # NOTE: This is a temporary solution and will be updated with cloud runner.
        from flagscale.auto_tuner.utils import is_master
        if is_master(config):
            tuner = AutoTuner(config)
            tuner.tune()
    else:
        if config.experiment.runner.get("type", "ssh") == "ssh":
            runner = SSHRunner(config)
        elif config.experiment.runner.get("type") == "cloud":
            runner = CloudRunner(config)
        else:
            raise ValueError(f"Unknown runner type {config.runner.type}")

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
