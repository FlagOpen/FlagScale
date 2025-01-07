import importlib
import uvicorn
# import networkx as nx
import matplotlib.pyplot as plt
import ray
from ray import workflow
import omegaconf
import logging as logger


from pydantic import create_model
from typing import Callable, Any
from fastapi import FastAPI, HTTPException, Request


class Builder:
    def __init__(self, config):
        self.config = config
        self.check_config(config)
        self.tasks = {}

    def check_config(self, config):
        if not config.get("deploy", None):
            raise ValueError("key deploy is missing for deployment configuration.")
        if not config.deploy.get("models", None):
            raise ValueError("key models is missing for building dag pipeline.")
    

    def find_final_node(self):
        whole_nodes = set(self.config["deploy"]["models"].keys())
        dependencies = set()

        for model_alias, model_config in self.config["deploy"]["models"].items():
            if len(model_config.get("depends", [])) > 0:
                dependencies.update(model_config.depends)
        
        output_node = whole_nodes - dependencies
        if len(output_node) != 1:
            raise ValueError(
                f"There should only have one final node but there are {len(output_node)} nodes {output_node}."
            )
        return list(output_node)[0]

    # def check_dag(self, visibilization=False):

    #     # Ensure that all dependencies are valid
    #     dag = {}
    #     for model_alias, model_config in self.config["deploy"]["models"].items():
    #         dependencies = []
    #         if "depends" in model_config:
    #             deps = model_config["depends"]
    #             if not isinstance(deps, (list, omegaconf.listconfig.ListConfig)):
    #                 deps = [deps]
    #             dependencies = deps
    #         else:
    #             dependencies = []
    #         dag[model_alias] = dependencies

    #         for dep in dependencies:
    #             if dep not in self.config["deploy"]["models"]:
    #                 raise ValueError(
    #                     f"Dependency {dep} for model {model_alias} not found in config['deploy']['models']"
    #                 )

    #     # Create a directed graph
    #     G = nx.DiGraph()

    #     # Add nodes and edges
    #     for node, neighbors in dag.items():
    #         for neighbor in neighbors:
    #             G.add_edge(node, neighbor)

    #     if visibilization or not nx.is_directed_acyclic_graph(G):
    #         pos = nx.spring_layout(G)
    #         plt.figure(figsize=(8, 6))
    #         nx.draw(
    #             G,
    #             pos,
    #             with_labels=True,
    #             node_color="lightblue",
    #             edge_color="gray",
    #             node_size=2000,
    #             font_size=15,
    #             arrows=True,
    #         )

    #         # Save the graph as an image without displaying it
    #         dag_file_name = "dag.png"
    #         plt.savefig(dag_file_name, bbox_inches="tight")
    #         plt.close()
    #         # Ensure that the graph is a DAG
    #         if not nx.is_directed_acyclic_graph(G):
    #             raise ValueError(
    #                 f"The graph contains cycles and is not a Directed Acyclic Graph (DAG). The dag can be visibilized at {dag_file_name}"
    #             )

    def build_task(self):
        ray.init(
            storage="/tmp/ray_workflow",
            runtime_env={
                "working_dir": self.config["root_path"],
            },
        )
        for model_alias, model_config in self.config["deploy"]["models"].items():
            module_name = model_config["module"]
            model_name = model_config["name"]
            module = importlib.import_module(module_name)
            model = getattr(module, model_name)
            num_gpus = model_config.resources.get("num_gpus", 0)
            self.tasks[model_alias] = ray.remote(model).options(num_gpus=num_gpus)
            # tasks[model_alias] = ray.remote(num_gpus=num_gpus)(model)
            # models[model_alias] = model
        # self.check_dag()
        return

    def run_task(self, *input_data):
        assert len(self.tasks) > 0
        models_to_process = list(self.config["deploy"]["models"].keys())

        model_nodes = {}

        while models_to_process:
            progress = False
            for model_alias in list(models_to_process):
                model_config = self.config["deploy"]["models"][model_alias]
                dependencies = []
                if "depends" in model_config:
                    deps = model_config["depends"]
                    if not isinstance(deps, (list, omegaconf.listconfig.ListConfig)):
                        deps = [deps]
                    dependencies = deps
                else:
                    dependencies = []

                if all(dep in model_nodes for dep in dependencies):
                    if dependencies:
                        if len(dependencies) > 1:
                            inputs = [model_nodes[dep] for dep in dependencies]
                            model_nodes[model_alias] = self.tasks[model_alias].bind(
                                *inputs
                            )
                        else:
                            model_nodes[model_alias] = self.tasks[model_alias].bind(
                                model_nodes[dependencies[0]]
                            )
                    else:
                        if len(input_data)==0:
                            model_nodes[model_alias] = self.tasks[model_alias].bind()
                        else:
                            model_nodes[model_alias] = self.tasks[model_alias].bind(
                                *input_data
                            )
                    models_to_process.remove(model_alias)
                    progress = True
            if not progress:
                raise ValueError("Circular dependency detected in model configuration")

        logger.info(f" =========== deploy model_nodes ============= ", model_nodes)
        find_final_node = self.find_final_node()

        final_node = model_nodes[find_final_node]
        ray.dag.vis_utils.plot(final_node, "output.jpg")
        final_result = workflow.run(final_node)
        return final_result

    def run_router_task(self, method="post"):

        router_config = self.config["deploy"].get("service")

        assert router_config and len(router_config) > 0
        name = router_config["name"]
        port = router_config["port"]
        request_names = router_config["request"]["names"]
        request_types = router_config["request"]["types"]

        RequestData = create_model(
            "Request",
            **{field: (type_, ...) for field, type_ in zip(request_names, request_types)},
        )
        app = FastAPI()

        if method.lower() == "post":

            @app.post(name)
            async def route_handler(request_data: RequestData):
                input_data = tuple(getattr(request_data, field) for field in request_names)
                try:
                    response = self.run_task(*input_data)
                    return response
                except Exception as e:
                    raise HTTPException(status_code=400, detail=str(e))

        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        uvicorn.run(app, host="127.0.0.1", port=port)
