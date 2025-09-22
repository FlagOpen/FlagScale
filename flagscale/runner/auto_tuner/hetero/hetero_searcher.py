import logging
from typing import Dict, List
from omegaconf import OmegaConf, DictConfig, ListConfig

from flagscale.runner.utils import parse_hostfile
from flagscale.runner.auto_tuner.search.searcher import Searcher

def _generate_layer_splits(n: int, k: int):
    """
    Helper function to solve the integer partitioning problem.
    Generates all possible ways to split n layers into k pipeline stages,
    where each stage must have at least one layer.
    """
    if k == 1:
        if n > 0:
            yield [n]
        return
    for i in range(1, n - k + 2):
        for rest in _generate_layer_splits(n - i, k - 1):
            # Ensure descending order to avoid duplicate partitions like [1, 2] and [2, 1].
            if i >= rest[0]:
                yield [i] + rest

class HeteroSearcher(Searcher):
    """
    A specialized searcher for heterogeneous environments.

    It discovers valid training strategies by searching through possible layer splits
    and resource allocations (process meshes), while respecting the physical boundaries
    of compute nodes. It supports a full range of tunable parameters, including
    parallelism, micro-batch size, and recomputation strategies.
    """
    def __init__(self, config: Dict, resources: Dict):
        """
        Initializes the HeteroSearcher.

        Args:
            config (Dict): The main configuration object.
            resources (Dict): A pre-parsed dictionary of cluster resources from a hostfile.
        """
        self.resources = resources
        super().__init__(config)

    def build_space(self, config: Dict) -> Dict:
        """
        Builds a comprehensive search space for heterogeneous tuning. It derives
        hardware information from the provided resources and parses hetero-specific
        and standard search parameters from the YAML configuration.
        """
        space = {}
        auto_tuner_config = config.experiment.auto_tuner
        hetero_space = auto_tuner_config.space

        # 1. Derive node-aware hardware information from the pre-parsed resources.
        self.node_info = []
        if not self.resources:
            raise ValueError("Heterogeneous tuning requires a valid hostfile/resources dict.")
        for _, resource_info in self.resources.items():
            self.node_info.append({
                "type": resource_info.get("type", "default_gpu"),
                "slots": resource_info.get("slots", 0)
            })
        self.logger.info(f"Received node-aware hardware info: {self.node_info}")

        # Helper to safely convert OmegaConf types to native Python types.
        def safe_to_container(value):
            if isinstance(value, (DictConfig, ListConfig)):
                return OmegaConf.to_container(value, resolve=True)
            return value

        # 2. Parse all search parameters using the safe converter.
        space["hetero_pipeline_layer_split"] = safe_to_container(hetero_space.get("hetero_pipeline_layer_split", "auto"))
        space["hetero_pp_size"] = safe_to_container(hetero_space.get("hetero_pp_size"))
        
        # CP and EP are constrained to 1 for now but are kept for future extensibility.
        space["hetero_cp_per_stage"] = [1]
        space["hetero_ep_per_stage"] = [1]
        
        # Parse standard tunable parameters
        space["micro_batch_size"] = safe_to_container(hetero_space.get("micro_batch_size", [1]))
        space["use_distributed_optimizer"] = safe_to_container(hetero_space.get("use_distributed_optimizer", [True, False]))
        space["sequence_parallel"] = safe_to_container(hetero_space.get("sequence_parallel", [True, False]))
        
        # Parse recomputation parameters
        space["use_recompute"] = safe_to_container(hetero_space.get("use_recompute", [True, False]))
        space["recompute_method"] = safe_to_container(hetero_space.get("recompute_method", ["uniform", "block"]))
        space["recompute_granularity"] = safe_to_container(hetero_space.get("recompute_granularity", ["full", "selective"]))
        num_layers = config.train.model.num_layers
        space["recompute_num_layers"] = safe_to_container(hetero_space.get("recompute_num_layers", list(range(1, num_layers + 1))))
        
        # Parse VPP parameter (fixed to 0/None as per constraint)
        space["num_layers_per_virtual_pipeline_stage"] = [0]
        
        # Ensure a default 'algo' config exists.
        if "algo" not in auto_tuner_config:
            auto_tuner_config.algo = {"name": "grid", "priority": None}
            
        return space

    def build_strategies(self, space: Dict, config: Dict) -> List[Dict]:
        """
        Builds all valid heterogeneous strategies by performing a Cartesian product
        across all defined dimensions.
        """
        self.logger.info("Building comprehensive heterogeneous strategies (Node-Aware)...")
        
        hetero_parallelism_part = self._generate_hetero_parallelism_part(space, config)
        self.logger.info(f"Generated {len(hetero_parallelism_part)} core heterogeneous parallelism strategies.")

        final_strategies = self._product_with_other_dims(hetero_parallelism_part, space, config)
        
        self.logger.info(f"Built a total of {len(final_strategies)} comprehensive candidate strategies.")
        return final_strategies

    def _generate_hetero_parallelism_part(self, space, config):
        """Generates the unique heterogeneous parallelism combinations."""
        strategies = []
        num_layers = config.train.model.num_layers
        cp_search_space = space["hetero_cp_per_stage"]
        ep_search_space = space["hetero_ep_per_stage"]
        mesh_combo_cache = {}

        if space["hetero_pipeline_layer_split"] == "auto":
            if not space.get("hetero_pp_size"):
                raise ValueError("In 'auto' mode, 'hetero_pp_size' must be provided.")
            for pp_size in space["hetero_pp_size"]:
                layer_splits = list(_generate_layer_splits(num_layers, pp_size))
                self._generate_strategies_for_pp_size(strategies, pp_size, layer_splits, config, cp_search_space, ep_search_space, mesh_combo_cache)
        else:
            splits_by_pp = {}
            for split in space["hetero_pipeline_layer_split"]:
                pp_size = len(split)
                if pp_size not in splits_by_pp: splits_by_pp[pp_size] = []
                splits_by_pp[pp_size].append(split)
            for pp_size, layer_splits in splits_by_pp.items():
                self._generate_strategies_for_pp_size(strategies, pp_size, layer_splits, config, cp_search_space, ep_search_space, mesh_combo_cache)

        return strategies

    def _generate_strategies_for_pp_size(self, strategies, pp_size, layer_splits, config, cp_search_space, ep_search_space, cache):
        """Helper to generate strategies for a specific pp_size."""
        self.logger.info(f"Processing {len(layer_splits)} layer split(s) for PP={pp_size}.")
        if pp_size not in cache:
            cache[pp_size] = self._find_node_aware_mesh_combinations(
                pp_stages_total=pp_size, nodes_available=[True] * len(self.node_info),
                config=config, cp_search_space=cp_search_space, ep_search_space=ep_search_space)
        mesh_combinations = cache[pp_size]
        self.logger.info(f"Found {len(mesh_combinations)} valid mesh combinations for PP={pp_size}.")
        
        for split in layer_splits:
            for mesh_combo in mesh_combinations:
                strategies.append({
                    "pipeline_model_parallel_size": pp_size,
                    "hetero_pipeline_layer_split": split,
                    "hetero_process_meshes": [item['mesh'] for item in mesh_combo],
                    "hetero_device_types": [item['device_type'] for item in mesh_combo],
                })

    def _product_with_other_dims(self, parallelism_part, space, config):
        """
        Performs a Cartesian product with other tunable dimensions and adds
        compatibility keys for base modules.
        """
        full_strategies = []
        for base_strategy in parallelism_part:
            meshes = base_strategy["hetero_process_meshes"]
            dp_list = [mesh[3] for mesh in meshes]
            base_dp = dp_list[0] if dp_list else 1

            for mbs in space["micro_batch_size"]:
                # Proactively prune based on the data validation rule.
                is_dp_compatible = all((base_dp * mbs) % dp == 0 for dp in dp_list)
                if not is_dp_compatible:
                    continue

                all_dp_are_one = all(dp == 1 for dp in dp_list)
                do_options = [False] if all_dp_are_one else space["use_distributed_optimizer"]
                for use_do in do_options:
                    for sp in space["sequence_parallel"]:
                        all_tp_gt_one = all(mesh[0] > 1 for mesh in meshes)
                        if sp and not all_tp_gt_one:
                            continue
                        for vpp_layers in space["num_layers_per_virtual_pipeline_stage"]:
                            for use_recompute in space["use_recompute"]:
                                strategy_keys = {
                                    **base_strategy,
                                    "micro_batch_size": mbs,
                                    "use_distributed_optimizer": use_do,
                                    "sequence_parallel": sp,
                                    "use_recompute": use_recompute,
                                    "num_layers_per_virtual_pipeline_stage": vpp_layers if vpp_layers > 0 else None,
                                    
                                    # Add compatibility keys for base Pruner/Generator modules
                                    "tensor_model_parallel_size": meshes[0][0],
                                    "context_parallel_size":      meshes[0][1],
                                    "expert_model_parallel_size": meshes[0][2],
                                    "data_parallel_size":         meshes[0][3],
                                    "decoder_first_pipeline_num_layers": None,
                                    "decoder_last_pipeline_num_layers": None,
                                }

                                if not use_recompute:
                                    strategy_keys.update({
                                        "recompute_method": None,
                                        "recompute_granularity": None,
                                        "recompute_num_layers": None,
                                    })
                                    full_strategies.append(strategy_keys)
                                else:
                                    for r_method in space["recompute_method"]:
                                        for r_granularity in space["recompute_granularity"]:
                                            for r_num_layers in space["recompute_num_layers"]:
                                                layers_per_stage = config.train.model.num_layers // base_strategy["pipeline_model_parallel_size"]
                                                if r_num_layers > layers_per_stage:
                                                    continue
                                                full_strategies.append({
                                                    **strategy_keys,
                                                    "recompute_method": r_method,
                                                    "recompute_granularity": r_granularity,
                                                    "recompute_num_layers": r_num_layers,
                                                })
        return full_strategies
    
    def _find_node_aware_mesh_combinations(self, pp_stages_total, nodes_available, config, cp_search_space, ep_search_space, current_path=None):
        """The core recursive, node-aware search algorithm."""
        if current_path is None: current_path = []
        if len(current_path) == pp_stages_total: return [current_path]
        
        results = []
        for i, is_available in enumerate(nodes_available):
            if is_available:
                node = self.node_info[i]
                gpus_on_node, device_type = node['slots'], node['type']
                
                for cp_size in cp_search_space:
                    for ep_size in ep_search_space:
                        if current_path and cp_size != current_path[0]['mesh'][1]: continue
                        if gpus_on_node % (cp_size * ep_size) != 0: continue
                        
                        tp_dp_product = gpus_on_node // (cp_size * ep_size)
                        for tp_size in range(1, tp_dp_product + 1):
                            if tp_dp_product % tp_size == 0:
                                dp_size = tp_dp_product // tp_size
                                if config.train.model.hidden_size % tp_size != 0: continue
                                
                                new_nodes_available = nodes_available[:]
                                new_nodes_available[i] = False
                                mesh = [tp_size, cp_size, ep_size, dp_size, 1]
                                new_path = current_path + [{'mesh': mesh, 'device_type': device_type}]
                                
                                results.extend(self._find_node_aware_mesh_combinations(
                                    pp_stages_total, new_nodes_available, config, cp_search_space, ep_search_space, new_path))
        return results
