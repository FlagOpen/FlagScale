import os
import sys
import threading
import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Optional, Union

import fastapi
import torch
import uvicorn
from benchmark_megatron_throughout import model_provider
from pydantic import BaseModel

pardir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(pardir, "megatron"))

from tools.stream_conversation.conversation_convo_v2 import (
    covert_prompt_to_input_ids_with_history,
)

from megatron import get_args, get_tokenizer, print_rank_0
from megatron.arguments import core_transformer_config_from_args
from megatron.checkpointing import load_checkpoint
from megatron.core import mpu
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.text_generation import generate_and_post_process
from megatron.training import get_model


class GenerateStatus(Enum):
    FAIL = 0
    SUCCESS = 1


class ArgsRequest(BaseModel):
    prompts: Union[str, list]
    temperature: Optional[float] = 0.9
    max_new_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9
    top_k_per_token: Optional[int] = 20
    seed: Optional[int] = 123
    sft: Optional[bool] = False
    template: Optional[str] = "aquila-legacy"
    history: Optional[list] = []
    max_gen_time: Optional[int] = 15
    # num_return_sequences: Optional[int] = 1
    # engine: str
    # echo_prompt: bool
    # stop_sequences: list


class ModelGeneration(metaclass=ABCMeta):
    def __init__(self, model=None, tokenizer=None):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._lock = threading.Lock()

    @abstractmethod
    async def generate(self, request: ArgsRequest):
        pass

    @abstractmethod
    async def post_process(self, data, request: ArgsRequest):
        pass

    async def infer(self, request: ArgsRequest):
        with self._lock:
            outputs = await self.generate(request)
            results = await self.post_process(outputs, request)
            return results


class AquilaGenerationBatch(ModelGeneration):
    def __init__(self, model=None, tokenizer=None):
        super().__init__(model, tokenizer)

    async def generate(self, request: ArgsRequest):
        prompts = request.prompts
        temperature = request.temperature
        max_new_tokens = request.max_new_tokens
        top_p = request.top_p
        top_k_per_token = request.top_k_per_token
        seed = request.seed
        sft = request.sft
        template = request.template
        history = request.history

        if template not in ["v1", "bair", "aquila-legacy"]:
            template = template

        if isinstance(prompts, str):
            prompts = [prompts]

        if max_new_tokens == 0:
            prompts = ["[CLS]"] + prompts

        if sft:
            new_prompts = []
            for p in prompts:
                p = covert_prompt_to_input_ids_with_history(
                    p, history, self._tokenizer, max_token=4096, template=template
                )
                new_prompts.append(p)
            prompts = new_prompts

        res_dict = {}
        try:
            choice = torch.cuda.LongTensor([1])
            torch.distributed.broadcast(choice, 0)
            responses, _, response_logprobs, _ = generate_and_post_process(
                model,
                prompts=prompts,
                tokens_to_generate=max_new_tokens,
                return_output_log_probs=True,
                top_k_sampling=top_k_per_token,
                top_p_sampling=top_p,
                top_p_decay=0.0,
                top_p_bound=0.0,
                temperature=temperature,
                add_BOS=False,
                use_eod_token_for_early_termination=True,
                stop_on_double_eol=False,
                stop_on_eol=False,
                prevent_newline_after_colon=False,
                random_seed=seed,
            )
            torch.cuda.empty_cache()
            res_dict["prompts"] = prompts
            res_dict["responses"] = responses
            res_dict["response_logprobs"] = response_logprobs
            res_dict["status"] = GenerateStatus.SUCCESS

        except Exception as e:
            print(f"Exception is {e}")
            print(f"occur a bug, please pay attention to it. return none result.")
            res_dict["prompts"] = prompts
            res_dict["responses"] = prompts
            res_dict["response_logprobs"] = 0.0
            res_dict["status"] = GenerateStatus.FAIL

        return res_dict

    async def post_process(self, output, request: ArgsRequest):
        def convert_ids_to_tokens(ids):
            convert_tokens = []
            vocab = self._tokenizer.vocab
            id2word = {v: k for k, v in vocab.items()}
            for t in ids:
                if t == 100006:
                    convert_tokens.append("[CLS]")
                else:
                    convert_tokens.append(id2word.get(t, "[unk]"))
            return convert_tokens

        completions = defaultdict(dict)
        max_input_length = 0
        max_output_length = 0
        if output["status"] == GenerateStatus.SUCCESS:
            for i, prompt in enumerate(output["prompts"]):
                max_input_length = max(
                    max_input_length, len(self._tokenizer.tokenize(prompt))
                )
                max_output_length = max(
                    max_output_length,
                    len(self._tokenizer.tokenize(output["responses"][i])),
                )

                response = output["responses"][i]
                response_logprob = output["response_logprobs"]

                if request.max_new_tokens != 0:
                    response = response[len(prompt) :]
                    if response.endswith("</s>"):
                        response_logprob = response_logprob[:-1]
                        response = response.replace("</s>", "")
                    ids = self._tokenizer.tokenize(response)
                    response_logprob = response_logprob[-len(ids) :]
                    if len(ids) == 0:
                        response_logprob = []
                    if len(response) > 0 and response[0] == " ":
                        response = response[1:]
                        response_logprob = response_logprob[1:]
                else:
                    ids = self._tokenizer.tokenize(response)
                    response_logprob = [0.0] + response_logprob
                convert_tokens = convert_ids_to_tokens(ids)

                completions[i]["text"] = response
                completions[i]["tokens"] = convert_tokens
                completions[i]["logprob"] = response_logprob
                tld = [{k: v} for k, v in zip(convert_tokens, response_logprob)]
                completions[i]["top_logprobs_dicts"] = tld
        else:
            for prompt in output["prompts"]:
                ids = self._tokenizer.tokenize(prompt)
                response_logprob = [0.0] * len(ids)
                convert_tokens = convert_ids_to_tokens(ids)

                completions[i]["text"] = prompt
                completions[i]["tokens"] = convert_tokens
                completions[i]["logprob"] = response_logprob
                tld = [{k: v} for k, v in zip(convert_tokens, response_logprob)]
                completions[i]["top_logprobs_dicts"] = tld

        return {
            "completions": completions,
            "max_input_length": max_input_length,
            "max_output_length": max_output_length,
        }


class AquilaGenerationStream(ModelGeneration):
    def __init__(self, model=None, tokenizer=None):
        super().__init__(model, tokenizer)

    async def generate(self, request: ArgsRequest):
        prompts = request.prompts
        temperature = request.temperature
        max_new_tokens = request.max_new_tokens
        top_p = request.top_p
        top_k_per_token = request.top_k_per_token
        seed = request.seed
        sft = request.sft
        template = request.template
        history = request.history

        if template not in ["v1", "bair", "aquila-legacy"]:
            template = "aquila-legacy"

        if seed > 0:
            torch.random.manual_seed(seed)

        assert type(prompts) is str
        if sft:
            prompts = covert_prompt_to_input_ids_with_history(
                prompts, history, self._tokenizer, 2048, template
            )

        prompts = [prompts]
        choice = torch.cuda.LongTensor([1])
        torch.distributed.broadcast(choice, 0)
        tokens = generate_and_post_process(
            model,
            prompts=prompts,
            tokens_to_generate=max_new_tokens,
            return_output_log_probs=True,
            top_k_sampling=top_k_per_token,
            top_p_sampling=top_p,
            top_p_decay=0.0,
            top_p_bound=0.0,
            temperature=temperature,
            add_BOS=False,
            use_eod_token_for_early_termination=True,
            stop_on_double_eol=False,
            stop_on_eol=False,
            prevent_newline_after_colon=False,
            random_seed=seed,
            stream=True,
        )
        torch.cuda.empty_cache()
        return {"tokens": tokens}

    async def post_process(self, output, request: ArgsRequest):
        def trans():
            s_time = time.time()
            while True:
                try:
                    yield next(output["tokens"])
                except Exception as e:
                    yield f"\nException is {e}"
                    break

                if time.time() - s_time > request.max_gen_time:
                    yield "\nTime UP"
                    break

        return fastapi.responses.StreamingResponse(trans(), media_type="text/plain")


class UvicornServer:
    def __init__(self, model):
        self._app = fastapi.FastAPI()
        self._model = model
        self._tokenizer = get_tokenizer()

    def run(self, **kwargs):

        func_types = ["Batch", "Stream"]
        router = fastapi.APIRouter()
        for func_type in func_types:
            model = eval("AquilaGeneration" + func_type)(self._model, self._tokenizer)
            router.add_api_route(
                "/Aquila/" + func_type,
                model.infer,
                methods=["post"],
                summary=f"{func_type}",
            )

        self._app.include_router(router)
        uvicorn.run(self._app, **kwargs)


if __name__ == "__main__":
    initialize_megatron(
        extra_args_provider=None,
        args_defaults={
            "tokenizer_type": "GPT2BPETokenizer",
            "no_load_rng": True,
            "no_load_optim": True,
        },
    )

    args = get_args()
    print(f"args is {args}")
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    print_rank_0(
        "WARNING: Forcing exit_on_missing_checkpoint to True for text " "generation."
    )
    args.exit_on_missing_checkpoint = True
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    print(f"get model successfully....")

    if args.load is not None:
        _ = load_checkpoint(model, None, None)
    print("loading model successfully....")

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        server_ins = UvicornServer(model)
        server_ins.run(host="0.0.0.0", port=5050, workers=1)

    while True:
        choice = torch.cuda.LongTensor(1)
        torch.distributed.broadcast(choice, 0)
        if choice[0].item() == 1:
            try:
                generate_and_post_process(model)
                torch.cuda.empty_cache()

            except ValueError as ve:
                pass
