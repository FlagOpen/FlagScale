import hydra
from omegaconf import DictConfig

from flagscale.runner.runner_train import SSHTrainRunner, CloudTrainRunner
from flagscale.runner.runner_inference import SSHInferenceRunner
from flagscale.runner.runner_serve import SSHServeRunner
from flagscale.runner.runner_compress import SSHCompressRunner


@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig) -> None:
    task_type = config.experiment.task.get("type", "train")
    if task_type == "train":
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
                runner = SSHTrainRunner(config)
            elif config.experiment.runner.get("type") == "cloud":
                runner = CloudTrainRunner(config)
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
    elif task_type == "inference":
        runner = SSHInferenceRunner(config)
        if config.action == "run":
            runner.run()
        elif config.action == "dryrun":
            runner.run(dryrun=True)
        elif config.action == "stop":
            runner.stop()
        else:
            raise ValueError(f"Unknown action {config.action}")
    elif task_type == "serve":
        runner = SSHServeRunner(config)
        if config.action == "run":
            runner.run()
        elif config.action == "test":
            runner.run(with_test=True)
    elif task_type == "compress":
        runner = SSHCompressRunner(config)
        if config.action == "run":
            runner.run()
        elif config.action == "dryrun":
            runner.run(dryrun=True)
        elif config.action == "stop":
            runner.stop()
        else:
            raise ValueError(f"Unknown action {config.action}")
    else:
        raise ValueError(f"Unknown task type {task_type}")


if __name__ == "__main__":
    main()
