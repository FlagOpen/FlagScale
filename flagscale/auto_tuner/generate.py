import os
import copy


class Generator:

    def __init__(self, config):
        self.config = config
        # TODO: Just a temporary solution, need to be configurated by user
        if "args_mapping" in config.experiment.auto_tuner:
            self.args_mapping = config.experiment.auto_tuner.args_mapping
        else:
            self.args_mapping = {
                "data_parallel_size": "data_parallel_size",
                "use_distributed_optimizer": "use_distributed_optimizer",
                "tensor_model_parallel_size": "tensor_model_parallel_size",
                "sequence_parallel": "sequence_parallel",
                "pipeline_model_parallel_size": "pipeline_model_parallel_size",
                "num_layers_per_virtual_pipeline_stage":
                "num_layers_per_virtual_pipeline_stage",
                "recompute_method": "recompute_method",
                "recompute_granularity": "recompute_granularity",
                "recompute_num_layers": "recompute_num_layers",
                "micro_batch_size": "micro_batch_size",
                "context_parallel_size": "context_parallel_size",
                "expert_model_parallel_size": "expert_model_parallel_size",
            }

    def _set_value(self, strategy, config):
        for key, value in self.args_mapping.items():
            if key in ["micro_batch_size"]:
                config.train.model[value] = strategy[key]
            elif key in ["data_parallel_size"]:
                continue
            else:
                if strategy[key] is None:
                    if value in config.train.system:
                        del config.train.system[value]
                    continue
                config.train.system[value] = strategy[key]

    def gen(self, strategy):
        config = copy.deepcopy(self.config)
        self._set_value(strategy, config)

        # Logging interval should be 1
        config.train.system.logging.log_interval = 1

        # Set redict and tee
        config.experiment.runner.tee = 3
        config.experiment.runner.redirects = 3

        # auto_tune should be true, it will not save ckpt when train ended and report memory every iteration
        config.train.system.auto_tune = True

        # Del lr_warmup_samples and train_samples to run megatron.
        assert "optimizer" in config.train.model
        assert "lr_scheduler" in config.train.model.optimizer
        if "lr_warmup_samples" in config.train.model.optimizer.lr_scheduler:
            del config.train.model.optimizer.lr_scheduler.lr_warmup_samples
        # Del lr_decay_samples and train_samples to run megatron.
        if "lr_decay_samples" in config.train.model.optimizer.lr_scheduler:
            del config.train.model.optimizer.lr_scheduler.lr_decay_samples
        # Del rampup_batch_size and train_samples to run megatron.
        if "rampup_batch_size" in config.train.model.optimizer.lr_scheduler:
            del config.train.model.optimizer.lr_scheduler.rampup_batch_size
        # Del lr_decay_samples and train_samples to run megatron.
        if "lr_warmup_fraction" in config.train.model.optimizer.lr_scheduler:
            del config.train.model.optimizer.lr_scheduler.lr_warmup_fraction

        if "train_samples" in config.train.model:
            del config.train.model.train_samples

        # Del checkpoint load
        if "checkpoint" in config.train.system:
            if "load" in config.train.system.checkpoint:
                del config.train.system.checkpoint.load
            if "save_interval" in config.train.system.checkpoint:
                config.train.system.checkpoint.save_interval = 2000

        # Set train_iters of each task
        if "control" in config.experiment.auto_tuner:
            config.train.model.train_iters = config.experiment.auto_tuner.control.get(
                "train_iters", 5)
        else:
            config.train.model.train_iters = 5

        # log dir
        config.experiment.exp_dir = os.path.join(config.experiment.exp_dir,
                                                 "auto_tuner",
                                                 f"task_{strategy['idx']}")

        return config

    def gen_best_task(self, strategy, config):
        self._set_value(strategy, config)
        return config
