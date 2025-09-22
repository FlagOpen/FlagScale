from flagscale.runner.auto_tuner.generate import Generator


class HeteroGenerator(Generator):
    """
    Generator for heterogeneous strategies. It translates an abstract strategy
    dictionary into a concrete, runnable OmegaConf object.
    """

    def __init__(self, config):
        super().__init__(config)

    def _set_value(self, strategy: dict, config: dict):
        """
        Overrides the base method to write all tunable parameters from the
        strategy into the configuration object.
        """
        super()._set_value(strategy, config)

        hetero_config = config.train.system.setdefault("hetero", {})
        hetero_config["enable_hetero"] = True
        hetero_config["hetero_pipeline_layer_split"] = strategy["hetero_pipeline_layer_split"]

        # [CORE MODIFICATION] Flatten the nested mesh list into a 1D list
        # before writing it to the final configuration.
        nested_meshes = strategy["hetero_process_meshes"]
        flat_meshes = [element for sublist in nested_meshes for element in sublist]
        hetero_config["hetero_process_meshes"] = flat_meshes

        hetero_config["hetero_device_types"] = strategy["hetero_device_types"]

        config.train.system["pipeline_model_parallel_size"] = strategy[
            "pipeline_model_parallel_size"
        ]
        config.train.system["sequence_parallel"] = strategy["sequence_parallel"]

    def gen(self, strategy: dict):
        return super().gen(strategy)

    def gen_best_task(self, strategy: dict, config: dict):
        self._set_value(strategy, config)
        return config
