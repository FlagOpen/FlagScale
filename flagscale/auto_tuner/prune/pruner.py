from .history import _HISTORY_BASED_PRUNE_FUNC


class Pruner:

    def __init__(self, config):
        self.config = config
        self.pruned_count = 0

    def prune(self, strategy, history=[]):
        """Prune strategy based on history recorded strategies."""
        not_run = False
        for func in _HISTORY_BASED_PRUNE_FUNC:
            if func(self.config, strategy, history):
                not_run = True
                break
        history.append(strategy)
        if not_run:
            self.pruned_count += 1
        return not_run
