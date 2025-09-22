import logging
from typing import Dict, List
import copy
from omegaconf import OmegaConf, DictConfig, ListConfig

from flagscale.runner.utils import parse_hostfile
from flagscale.runner.auto_tuner.search.searcher import Searcher

def _generate_layer_splits(n: int, k: int):
    """
    Helper function to solve the integer partitioning problem.
    """
    if k == 1:
        if n > 0:
            yield [n]
        return
    for i in range(1, n - k + 2):
        for rest in _generate_layer_splits(n - i, k - 1):
            if i >= rest[0]:
                yield [i] + rest

class HeteroSearcher(Searcher):
    """
    A specialized searcher for heterogeneous environments that interprets a
    flexible mesh template from the user configuration to generate strategies.
    """
    def __init__(self, config: Dict, resources: Dict):
        self.resources = resources
        super().__init__(config)

    def build_space(self, config: Dict) -> Dict:
        """
        Builds the search space by parsing the mesh template and other dimensions.
        """
        space = {}
        auto_tuner_config = config.experiment.auto_tuner
        hetero_space = auto_tuner_config.space

        self.node_info = []
        if not self.resources:
            raise ValueError("Heterogeneous tuning requires a valid hostfile/resources dict.")
        for _, resource_info in self.resources.items():
            self.node_info.append({
                "type": resource_info.get("type", "default_gpu"),
                "slots": resource_info.get("slots", 0)
            })
        self.logger.info(f"Received node-aware hardware info: {self.node_info}")

        def safe_to_container(value):
            if isinstance(value, (DictConfig, ListConfig)):
                return OmegaConf.to_container(value, resolve=True)
            return value

        mesh_template_raw = hetero_space.get("hetero_process_meshes")
        if not mesh_template_raw:
            raise ValueError("'hetero_process_meshes' must be defined for heterogeneous tuning.")
        
        mesh_template = safe_to_container(mesh_template_raw)
        if len(mesh_template) % 5 != 0:
            raise ValueError(f"'hetero_process_meshes' length must be a multiple of 5, got {len(mesh_template)}.")
        
        self.pp_size = len(mesh_template) // 5
        self.mesh_template = [mesh_template[i:i+5] for i in range(0, len(mesh_template), 5)]
        self.logger.info(f"Inferred pp_size={self.pp_size} from mesh template: {self.mesh_template}")

        space["hetero_pipeline_layer_split"] = safe_to_container(hetero_space.get("hetero_pipeline_layer_split", "auto"))
        space["micro_batch_size"] = safe_to_container(hetero_space.get("micro_batch_size", [1]))
        space["use_distributed_optimizer"] = safe_to_container(hetero_space.get("use_distributed_optimizer", [True, False]))
        space["sequence_parallel"] = safe_to_container(hetero_space.get("sequence_parallel", [True, False]))
        space["use_recompute"] = safe_to_container(hetero_space.get("use_recompute", [True, False]))
        space["recompute_method"] = safe_to_container(hetero_space.get("recompute_method", ["uniform", "block"]))
        space["recompute_granularity"] = safe_to_container(hetero_space.get("recompute_granularity", ["full", "selective"]))
        num_layers = config.train.model.num_layers
        space["recompute_num_layers"] = safe_to_container(hetero_space.get("recompute_num_layers", list(range(1, num_layers + 1))))
        space["num_layers_per_virtual_pipeline_stage"] = [0]
        
        if "algo" not in auto_tuner_config:
            auto_tuner_config.algo = {"name": "grid", "priority": None}
            
        return space

    def build_strategies(self, space: Dict, config: Dict) -> List[Dict]:
        """Builds all valid strategies based on the parsed mesh template."""
        self.logger.info("Building comprehensive heterogeneous strategies from template...")
        
        hetero_parallelism_part = self._generate_hetero_parallelism_part(space, config)
        self.logger.info(f"Generated {len(hetero_parallelism_part)} core heterogeneous parallelism strategies.")

        filtered_parallelism_part = self._filter_parallelism_strategies(hetero_parallelism_part, config)
        self.logger.info(f"Filtered down to {len(filtered_parallelism_part)} valid parallelism strategies.")

        final_strategies = self._product_with_other_dims(filtered_parallelism_part, space, config)
        self.logger.info(f"Built a total of {len(final_strategies)} comprehensive candidate strategies.")
        return final_strategies

    def _generate_hetero_parallelism_part(self, space, config):
        """Generates parallelism combinations from the mesh template."""
        strategies = []
        num_layers = config.train.model.num_layers
        
        if space["hetero_pipeline_layer_split"] == "auto":
            layer_splits = list(_generate_layer_splits(num_layers, self.pp_size))
        else:
            layer_splits = space["hetero_pipeline_layer_split"]
        
        mesh_combinations = self._find_node_aware_mesh_combinations(
            pp_stages_total=self.pp_size,
            nodes_available=[True] * len(self.node_info),
            config=config,
            mesh_template=self.mesh_template
        )
        self.logger.info(f"Found {len(mesh_combinations)} valid mesh instantiations for the template.")
        
        for split in layer_splits:
            for mesh_combo in mesh_combinations:
                # [CORE MODIFICATION] Keep meshes in their nested list format internally.
                strategies.append({
                    "pipeline_model_parallel_size": self.pp_size,
                    "hetero_pipeline_layer_split": split,
                    "hetero_process_meshes": [item['mesh'] for item in mesh_combo], # <-- Keep as nested list
                    "hetero_device_types": [item['device_type'] for item in mesh_combo],
                })
        return strategies

    def _filter_parallelism_strategies(self, strategies: List[Dict], config: Dict) -> List[Dict]:
        """Applies validation rules to filter out invalid strategies."""
        valid_strategies = []
        for strat in strategies:
            meshes = strat["hetero_process_meshes"] # Expects a nested list
            if not config.train.model.get("untie_embeddings_and_output_weights", False):
                if meshes and meshes[0][0] != meshes[-1][0]:
                    continue
            valid_strategies.append(strat)
        return valid_strategies

    def _product_with_other_dims(self, parallelism_part: List[Dict], space: Dict, config: Dict) -> List[Dict]:
        """Performs a Cartesian product with other dimensions and adds compatibility keys."""
        full_strategies = []
        for base_strategy in parallelism_part:
            meshes = base_strategy["hetero_process_meshes"] # Expects a nested list
            dp_list = [mesh[3] for mesh in meshes]
            base_dp = dp_list[0] if dp_list else 1
            for mbs in space["micro_batch_size"]:
                if not all((base_dp * mbs) % dp == 0 for dp in dp_list):
                    continue
                all_dp_are_one = all(dp == 1 for dp in dp_list)
                do_options = [False] if all_dp_are_one else space["use_distributed_optimizer"]
                for use_do in do_options:
                    for sp in space["sequence_parallel"]:
                        if sp and not all(mesh[0] > 1 for mesh in meshes):
                            continue
                        vpp_layers = space["num_layers_per_virtual_pipeline_stage"][0]
                        for use_recompute in space["use_recompute"]:
                            strategy_keys = {
                                **base_strategy, "micro_batch_size": mbs, "use_distributed_optimizer": use_do,
                                "sequence_parallel": sp, "use_recompute": use_recompute,
                                "num_layers_per_virtual_pipeline_stage": vpp_layers if vpp_layers > 0 else None,
                                "tensor_model_parallel_size": meshes[0][0], "context_parallel_size": meshes[0][1],
                                "expert_model_parallel_size": meshes[0][2], "data_parallel_size": meshes[0][3],
                                "decoder_first_pipeline_num_layers": None, "decoder_last_pipeline_num_layers": None,
                            }
                            if not use_recompute:
                                strategy_keys.update({"recompute_method": None, "recompute_granularity": None, "recompute_num_layers": None})
                                full_strategies.append(strategy_keys)
                            else:
                                for r_method in space["recompute_method"]:
                                    for r_granularity in space["recompute_granularity"]:
                                        for r_num_layers in space["recompute_num_layers"]:
                                            layers_per_stage = config.train.model.num_layers // base_strategy["pipeline_model_parallel_size"]
                                            if r_num_layers > layers_per_stage: continue
                                            full_strategies.append({**strategy_keys, "recompute_method": r_method, "recompute_granularity": r_granularity, "recompute_num_layers": r_num_layers})
        return full_strategies
    
    def _find_node_aware_mesh_combinations(self, pp_stages_total, nodes_available, config, mesh_template, current_path=None):
        """Recursively finds all valid process mesh instantiations that satisfy the template."""
        if current_path is None: current_path = []
        if len(current_path) == pp_stages_total: return [current_path]
        
        results = []
        current_stage_idx = len(current_path)
        stage_template = mesh_template[current_stage_idx]
        
        for i, is_available in enumerate(nodes_available):
            if is_available:
                node = self.node_info[i]
                gpus_on_node, device_type = node['slots'], node['type']
                possible_meshes = self._get_possible_meshes_for_node(gpus_on_node, stage_template, config)
                for mesh in possible_meshes:
                    new_nodes_available = nodes_available[:]
                    new_nodes_available[i] = False
                    new_path = current_path + [{'mesh': mesh, 'device_type': device_type}]
                    results.extend(self._find_node_aware_mesh_combinations(
                        pp_stages_total, new_nodes_available, config, mesh_template, new_path))
        return results

    def _get_possible_meshes_for_node(self, gpus_on_node, template, config):
        """Finds all valid mesh fillings for a given node and template."""
        possible_meshes = []
        tp_t, cp_t, ep_t, dp_t, pp_t = template

        def get_range(template_value, max_value):
            if isinstance(template_value, str) and template_value.lower() == 'auto':
                return range(1, max_value + 1)
            elif isinstance(template_value, list):
                return template_value
            elif isinstance(template_value, int):
                return [template_value]
            raise TypeError(f"Unsupported type in mesh template: {type(template_value)}")

        tp_range, cp_range, ep_range, dp_range, pp_range = (get_range(t, gpus_on_node) for t in template)

        for tp in tp_range:
            for cp in cp_range:
                for ep in ep_range:
                    for dp in dp_range:
                        for pp in pp_range:
                            if tp * cp * ep * dp * pp != gpus_on_node: continue
                            if config.train.model.hidden_size % tp != 0: continue
                            possible_meshes.append([tp, cp, ep, dp, pp])
        return possible_meshes
