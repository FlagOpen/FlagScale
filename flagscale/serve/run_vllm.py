import asyncio
import json
import logging
import sys
import time
from typing import Any, AsyncGenerator, Optional

import ray
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
)
from vllm.utils import random_uuid

# Compatible with both command-line execution and source code execution.
try:
    import flag_scale
except Exception as e:
    pass

from flagscale import serve

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


app = FastAPI()

from ray import serve

# ray.init(log_to_driver=True, configure_logging=True, logging_level=logging.INFO)
logger = logging.getLogger("ray.serve")

logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("serve.log"))


@serve.deployment(**get_deploy_config("LLMActor"))
class LLMActor:
    def __init__(self):
        engine_args = AsyncEngineArgs(**get_model_config(self.__class__.__name__))
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def generate(self, prompt, sampling_params, request_id):
        return self.engine.generate(prompt, sampling_params, request_id)


@serve.deployment(num_replicas="auto")
@serve.ingress(app)
class LLMService:
    def __init__(self, llm_actor):
        self.llm_actor = llm_actor

    @app.post("/v1/completions")
    async def generate_handler(self, request: CompletionRequest):
        print("receive request ==============", request, flush=True)
        logger.info(f"Received request --------------- {request}")
        # request = await req.json()
        prompt = request.prompt
        stream = request.stream
        request_id = "cmpl-" + random_uuid()
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        results_generator = self.llm_actor.generate.options(stream=True).remote(
            prompt,
            sampling_params,
            request_id,
        )

        if stream:
            # In streaming mode, retrieve tokens from the LLMActor.
            async def stream_results() -> AsyncGenerator[bytes, None]:
                async for request_output in results_generator:
                    prompt = request_output.prompt
                    assert prompt is not None
                    text_outputs = "".join(
                        output.text for output in request_output.outputs
                    )

                    chunk = CompletionStreamResponse(
                        id=request_id,
                        created=int(time.time()),
                        model=request.model,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=0,
                                text=text_outputs,
                                logprobs=None,
                                finish_reason=None,
                                stop_reason=None,
                            )
                        ],
                    )
                    response_json = chunk.model_dump_json(exclude_unset=False)
                    yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

                # yield (json.dumps(ret) + "\n").encode("utf-8")

            return StreamingResponse(stream_results(), media_type="text/event-stream")
        else:
            # Non-streaming mode: call the regular generate method.

            final_output = None
            try:
                async for request_output in results_generator:
                    final_output = request_output
            except asyncio.CancelledError:
                return Response(status_code=499)

            assert final_output is not None
            prompt = final_output.prompt
            assert prompt is not None
            text_outputs = "".join(output.text for output in final_output.outputs)

            ret = CompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    {
                        "text": text_outputs,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                usage={
                    "prompt_tokens": len(request.prompt.split()),
                    "completion_tokens": len(text_outputs.split()),
                    "total_tokens": len(request.prompt.split())
                    + len(text_outputs.split()),
                },
            )

            return JSONResponse(content=ret.model_dump())

    @app.post("/v1/chat/completions")
    async def generate_handler(self, request: ChatCompletionRequest):
        print("receive request ==============", request, flush=True)
        logger.info(f"Received request --------------- {request}")
        user_message = request.messages[-1]["content"]
        stream = request.stream
        request_id = "cmpl-" + random_uuid()
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        results_generator = self.llm_actor.generate.options(stream=True).remote(
            user_message,
            sampling_params,
            request_id,
        )

        if stream:
            # In streaming mode, retrieve tokens from the LLMActor.
            async def stream_results() -> AsyncGenerator[bytes, None]:
                async for request_output in results_generator:
                    prompt = request_output.prompt
                    assert prompt is not None
                    text_outputs = "".join(
                        output.text for output in request_output.outputs
                    )

                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=int(time.time()),
                        model=request.model,
                        choices=[
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": text_outputs},
                                "logprobs": None,
                                "finish_reason": None,
                                "stop_reason": None,
                            }
                        ],
                    )
                    response_json = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

                # yield (json.dumps(ret) + "\n").encode("utf-8")

            return StreamingResponse(stream_results(), media_type="text/event-stream")
        else:
            # Non-streaming mode: call the regular generate method.

            final_output = None
            try:
                async for request_output in results_generator:
                    final_output = request_output
            except asyncio.CancelledError:
                return Response(status_code=499)

            assert final_output is not None
            prompt = final_output.prompt
            assert prompt is not None
            text_outputs = "".join(output.text for output in final_output.outputs)

            ret = ChatCompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text_outputs},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                usage={
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(text_outputs.split()),
                    "total_tokens": len(user_message.split())
                    + len(text_outputs.split()),
                },
            )

            return JSONResponse(content=ret.model_dump())


if __name__ == "__main__":
    serve.start(
        http_options={"host": "0.0.0.0", "port": TASK_CONFIG.serve.deploy.service.port}
    )
    llm_actor = LLMActor.bind()
    serve.run(
        LLMService.bind(llm_actor),
        name="vllm_service",
        route_prefix="/",
        blocking=True,
    )
