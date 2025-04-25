import logging

logger = logging.getLogger("FlagScale-AutoTuner")


def prune_by_memory_model(config, strategy, history=[]):
    if "memory_model" in strategy:
        if strategy["memory_model"] > config.experiment.auto_tuner.memory_model.gpu_memory:
            logger.info(
                f"The strategy {strategy} has been pruned by modeling memory {int(strategy['memory_model'])}MB."
            )
            strategy["max_mem"] = "OOM"
            strategy["performance"] = None
            strategy["pruned"] = True
            return True
    return False


def prune_by_memory_model_util(config, strategy, history=[]):
    if "memory_model" in strategy:
        if (
            strategy["memory_model"] > config.experiment.auto_tuner.memory_model.gpu_memory * 0.8
            or strategy["memory_model"] < config.experiment.auto_tuner.memory_model.gpu_memory * 0.2
        ):
            logger.info(
                f"The strategy {strategy} has been pruned by modeling memory util and its memory is {int(strategy['memory_model'])}MB."
            )
            strategy["max_mem"] = None
            strategy["performance"] = None
            strategy["pruned"] = True
            return True
    return False
