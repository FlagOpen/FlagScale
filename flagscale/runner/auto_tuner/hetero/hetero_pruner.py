import logging

from flagscale.runner.auto_tuner.prune.history import (
    _HISTORY_BASED_PRUNE_FUNC,
    prune_by_sequence_parallel as original_prune_by_sp,
)
from flagscale.runner.auto_tuner.prune.memory import (
    prune_by_memory_model,
    prune_by_memory_model_util,
)
from flagscale.runner.auto_tuner.prune.pruner import Pruner
from flagscale.runner.auto_tuner.utils import beside, compare_by_recompute


class HeteroPruner(Pruner):
    """
    Pruner for heterogeneous strategies. Re-implements the pruning loop
    to selectively override specific base history rules (like sequence parallel)
    while reusing others directly.
    """

    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger("FlagScale-AutoTuner")

    # prune_by_meshes_mbs_recompute_oom method
    # Ensure this method does NOT increment self.pruned_count itself.
    def prune_by_meshes_mbs_recompute_oom(self, strategy: dict, history: list) -> bool:
        """
        Checks if pruning is needed based on identical meshes/mbs OOM history.
        Marks the strategy if pruning is needed and returns True, otherwise False.
        """
        current_meshes = strategy.get('hetero_process_meshes')
        current_mbs = strategy.get('micro_batch_size')
        if not current_meshes:
            return False
        for item in history:
            if item.get('max_mem') == 'OOM':
                item_meshes = item.get('hetero_process_meshes')
                if item_meshes == current_meshes:
                    item_mbs = item.get('micro_batch_size')
                    if item_mbs == current_mbs:
                        if compare_by_recompute(strategy, item):
                            reason = (
                                f"Identical meshes/mbs OOM: History task {item.get('idx', 'N/A')} "
                                f"OOM'd with same meshes/mbs and <= demanding recompute."
                            )
                            self.logger.info(f"Strategy {strategy.get('idx', 'N/A')}: {reason}")
                            self._mark_as_pruned(strategy, reason)
                            strategy['max_mem'] = 'OOM_predicted_meshes_mbs_recompute'
                            strategy['performance'] = None
                            # NO pruned_count increment here
                            return True  # Signal pruning
        return False  # No pruning

    # Define the corrected SP pruning logic internally ---
    def _corrected_prune_by_sequence_parallel(self, config, strategy, history=[]):
        """
        Internal version of prune_by_sequence_parallel with correction for all TP=1 case.
        Mimics signature and side effects of the original function.
        """
        sequence_parallel = strategy.get("sequence_parallel", False)

        all_tp_are_one = False
        meshes = strategy.get("hetero_process_meshes")
        if meshes:
            # Direct check is safest
            all_tp_are_one = all(mesh[0] == 1 for mesh in meshes)

        retrieval = beside(["sequence_parallel"], strategy, history)
        if retrieval:
            for item in retrieval:
                # Performance prune: ONLY apply if NOT all TPs are one
                if not all_tp_are_one:
                    if (
                        item.get("sequence_parallel")
                        and item.get("performance") is not None
                        and not sequence_parallel
                    ):
                        reason = "Pruned by sequence_parallel performance (TP>1)"
                        self.logger.info(f"Strategy {strategy.get('idx', 'N/A')}: {reason}.")
                        strategy["performance"] = item["performance"]
                        strategy["max_mem"] = item["max_mem"]
                        strategy["pruned"] = True
                        strategy["prune_reason"] = reason
                        return True  # Prune

                # Memory prune
                if (
                    item.get("sequence_parallel")
                    and item.get("max_mem") == "OOM"
                    and not sequence_parallel
                ):
                    reason = "Pruned by sequence_parallel memory"
                    self.logger.info(f"Strategy {strategy.get('idx', 'N/A')}: {reason}.")
                    strategy["max_mem"] = "OOM"
                    strategy["performance"] = None
                    strategy["pruned"] = True
                    strategy["prune_reason"] = reason
                    return True  # Prune
        return False  # Do not prune

    def prune(self, strategy: dict, history: list = None) -> bool:
        """
        Determines if a given strategy should be pruned using hetero-specific validation,
        static memory checks, hetero-specific history rules, and general history rules
        with selective override for sequence parallel logic.
        """
        if history is None:
            history = []

        # 1. Hetero-specific validity check
        if not self._is_strategy_valid(strategy):
            self._mark_as_pruned(strategy, 'Invalid Hetero Configuration')
            history.append(strategy)
            self.pruned_count += 1
            return True

        not_run = False  # Flag to track pruning decision

        # 2. Hetero-specific OOM prediction (marks strategy if prunes)
        if self.prune_by_meshes_mbs_recompute_oom(strategy, history):
            not_run = True

        # 3. Static memory model checks (only if not already pruned)
        if not not_run and "memory_model" in self.config.experiment.auto_tuner:
            # These functions mark strategy and update pruned_by_memory_model count
            if prune_by_memory_model(self.config, strategy, history):
                not_run = True
                self.pruned_by_memory_model += 1
            elif prune_by_memory_model_util(self.config, strategy, history):
                not_run = True
                self.pruned_by_memory_model += 1

        # 4. History-based checks using ORIGINAL list, but INTERCEPTING sp rule
        #    Run only if not already marked for pruning
        if not not_run:
            # Iterate through the ORIGINAL list from history.py
            for func in _HISTORY_BASED_PRUNE_FUNC:
                # Check if the current function is the one we want to replace
                if (
                    func.__name__ == original_prune_by_sp.__name__
                ):  # Compare by name or function object
                    # Call our corrected version instead
                    if self._corrected_prune_by_sequence_parallel(self.config, strategy, history):
                        not_run = True  # Our version decided to prune
                        # Our version already marked the strategy
                        break  # Stop checking other history rules
                else:
                    # Call the original function from history.py directly
                    if func(self.config, strategy, history):
                        not_run = True  # Original function decided to prune
                        # Original function already marked the strategy
                        break  # Stop checking other history rules

        # 5. Final handling (Matches parent Pruner.prune end logic)
        history.append(strategy)
        if not_run:
            self.pruned_count += 1

        return not_run

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
