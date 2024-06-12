import logging

logger = logging.getLogger("FlagScale-AutoTuner")


def prune_by_memory_model(config, strategy, history=[]):
    if "modeling_memory" in strategy:
        if (
            strategy["modeling_memory"]
            > config.experiment.auto_tuner.memory_model.gpu_memory
        ):
            logger.info(
                f"The strategy {strategy} has been pruned by modeling memory {int(strategy['modeling_memory'])}MB."
            )
            strategy["max_mem"] = "OOM"
            strategy["performance"] = None
            strategy["pruned"] = True
            return True
    return False


def prune_by_memory_model_util(config, strategy, history=[]):
    if "modeling_memory" in strategy:
        if (
            strategy["modeling_memory"]
            > config.experiment.auto_tuner.memory_model.gpu_memory * 0.8
            or strategy["modeling_memory"]
            < config.experiment.auto_tuner.memory_model.gpu_memory * 0.2
        ):
            logger.info(
                f"The strategy {strategy} has been pruned by modeling memory util and its memory is {int(strategy['modeling_memory'])}MB."
            )
            strategy["max_mem"] = None
            strategy["performance"] = None
            strategy["pruned"] = True
            return True
    return False
