import logging

from flagscale.runner.auto_tuner.prune.pruner import Pruner


class HeteroPruner(Pruner):
    """
    Pruner for heterogeneous strategies. It applies hetero-specific validation
    and then delegates to the base Pruner for generic, history-based rules.
    """

    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger("FlagScale-AutoTuner")

    def prune(self, strategy: dict, history: list = None) -> bool:
        """
        Determines if a given strategy should be pruned.
        """
        if history is None:
            history = []
        # First, Heterogeneous-Specific Pruning Rules ---
        if not self._is_strategy_valid(strategy):
            self._mark_as_pruned(strategy, 'Invalid Hetero Configuration')
            history.append(strategy)
            self.pruned_count += 1
            return True

        # Then, delegate to the parent class for generic pruning (e.g., memory model).
        return super().prune(strategy, history)

    def _mark_as_pruned(self, strategy: dict, reason: str):
        """Helper to mark a strategy as pruned."""
        strategy['pruned'] = True
        strategy['prune_reason'] = reason

    def _is_strategy_valid(self, strategy: dict) -> bool:
        """Checks a strategy's validity against key architectural constraints."""
        required_keys = [
            "pipeline_model_parallel_size",
            "hetero_pipeline_layer_split",
            "hetero_process_meshes",
            "hetero_device_types",
        ]
        if not all(key in strategy for key in required_keys):
            return False

        pp_size = strategy["pipeline_model_parallel_size"]
        layer_split = strategy["hetero_pipeline_layer_split"]
        meshes = strategy["hetero_process_meshes"]

        if len(layer_split) != pp_size:
            return False
        if sum(layer_split) != self.config.train.model.num_layers:
            return False

        untie_embeddings = self.config.train.model.get("untie_embeddings_and_output_weights", False)
        if not untie_embeddings and meshes and meshes[0][0] != meshes[-1][0]:
            return False

        return True
