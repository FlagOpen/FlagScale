import logging

from ..prune.pruner import Pruner


class HeteroPruner(Pruner):

    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger("FlagScale-AutoTuner")
        self.logger.info(
            "HeteroPruner initialized in pass-through mode (V1). No strategies will be pruned."
        )

    def prune(self, strategy, history=None):

        return False
