import logging
import sys
import time

import ray
from fastapi import FastAPI
from pydantic import BaseModel

from vllm import LLM, SamplingParams

# Compatible with both command-line execution and source code execution.
try:
    import flag_scale
except Exception as e:
    pass

from flagscale import serve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("serve.log"), logging.StreamHandler(sys.stdout)],
)

serve.load_args()
TASK_CONFIG = serve.task_config


def get_model_config(model_name):
    if not TASK_CONFIG.get("serve", None):
        raise ValueError("No 'serve' section found in task config.")
    if not TASK_CONFIG.serve.get("model_args", None):
        raise ValueError("No 'model_args' section found in task config.")
    model_config = TASK_CONFIG.serve.model_args.get(model_name, None)

    if model_config:
        return model_config
    else:
        raise ValueError(f"No model config found for model {model_name}.")


def get_deploy_config(model_name):
    if not TASK_CONFIG.get("serve", None):
        raise ValueError("No 'serve' section found in task config.")
    if not TASK_CONFIG.serve.get("deploy", None):
        raise ValueError("No 'deploy' section found in task config.")
    resource_config = {}

    if TASK_CONFIG.serve.deploy.get(
        "models", None
    ) and TASK_CONFIG.serve.deploy.models.get(model_name, None):
        models_resource_config = TASK_CONFIG.serve.deploy.models.get(model_name, None)
        ray_actor_options = {}
        resource_set = {"num_gpus", "num_cpus"}
        for item in resource_set:
            if item in models_resource_config:
                ray_actor_options[item] = models_resource_config[item]
        if ray_actor_options:
            resource_config["ray_actor_options"] = ray_actor_options
        if "num_replicas" in models_resource_config:
            resource_config["num_replicas"] = models_resource_config["num_replicas"]

    if not resource_config:
        resource_config = {"num_replicas": 1}
    return resource_config


class OpenAIRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 32000
    temperature: float = 1.0
    top_p: float = 1.0


app = FastAPI()

from ray import serve

print(" config =============================== ", TASK_CONFIG, flush=True)
print(
    "model resource ========================= ",
    get_deploy_config("LLMActor"),
    flush=True,
)


@serve.deployment(**get_deploy_config("LLMActor"))
class LLMActor:
    def __init__(self):
        self.llm = LLM(**get_model_config("LLMActor"))

    def generate(self, prompt, max_tokens, temperature, top_p):
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        return self.llm.generate(prompt, sampling_params)


@serve.deployment
@serve.ingress(app)
class LLMService:
    def __init__(self, llm_actor):
        self.llm_actor = llm_actor

    @app.post("/v1/completions")
    async def generate_handler(self, request: OpenAIRequest):
        # Generate through separate LLM deployment
        generation_result = await self.llm_actor.generate.remote(
            request.prompt, request.max_tokens, request.temperature, request.top_p
        )

        # Format the response in OpenAI style
        response = {
            "id": "cmpl-1234567890",  # You can generate a unique ID here
            "object": "text_completion",
            "created": int(time.time()),  # Placeholder for timestamp
            "model": request.model,
            "choices": [
                {
                    "text": generation_result[0].outputs[0].text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(generation_result[0].outputs[0].text.split()),
                "total_tokens": len(request.prompt.split())
                + len(generation_result[0].outputs[0].text.split()),
            },
        }

        return response


if __name__ == "__main__":
    serve.start(
        http_options={"host": "0.0.0.0", "port": TASK_CONFIG.serve.deploy.service.port}
    )
    llm_actor = LLMActor.bind()
    serve.run(
        LLMService.bind(llm_actor),
        name="llm_service",
        route_prefix="/",
        blocking=True,
    )
