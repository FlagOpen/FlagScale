from omegaconf import open_dict

from ..generate import Generator


class HeteroGenerator(Generator):

    def __init__(self, config):
        super().__init__(config)
        hetero_mapping = {
            "hetero_pipeline_layer_split": "hetero_pipeline_layer_split",
            "hetero_process_meshes": "hetero_process_meshes",
            "hetero_device_types": "hetero_device_types",
        }
        self.args_mapping.update(hetero_mapping)

    def _set_value(self, strategy, config):
        with open_dict(config):
            if "hetero" not in config.train.system:
                config.train.system.hetero = {}

            config.train.system.hetero.enable_hetero = True

            for key, value_path in self.args_mapping.items():
                if key not in strategy:
                    continue

                if key.startswith("hetero_"):
                    config.train.system.hetero[value_path] = strategy[key]
                elif key == "micro_batch_size":
                    config.train.model[value_path] = strategy[key]
                elif key == "data_parallel_size":
                    continue
                else:
                    if strategy[key] is None:
                        if value_path in config.train.system:
                            del config.train.system[value_path]
                        continue
                    config.train.system[value_path] = strategy[key]
