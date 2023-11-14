# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import socket
from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
import os
import uvicorn, json, datetime
import json
from asgiref.sync import sync_to_async
from megatron.text_generation.api_single_thread import generate_and_post_process_single_thread
import time
import torch
import threading
import random
import sys
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
import asyncio
from tools.stream_conversation.conversation_convo_v2 import covert_prompt_to_input_ids_with_history


from datetime import timedelta

try:
    import torch_xmlir
    from hyperparameter import param_scope
except:
    torch_xmlir = None

gloo_group = None

def get_tokenizer():
    from megatron.tokenizer.tokenizer import _AquilaTokenizer
    vocab_file = "examples/aquila/tokenizer/vocab.json"
    merge_file = "examples/aquila/tokenizer/merges.txt"
    special = "examples/aquila/tokenizer/special_tokens.txt"
    tokenizer = _AquilaTokenizer(vocab_file, merge_file, special)

    return tokenizer

tokenizer = get_tokenizer()
lock = threading.Lock()
lock_stream = asyncio.Lock()

vocab = tokenizer.vocab

id2word = {v:k for k, v in vocab.items()}

from tools.conversation import get_prompt
def make_sft_prompts(prompts):
    new_prompts = []
    for p in prompts:
        p = get_prompt(p)
        new_prompts.append(p)
    return new_prompts

class UvicornServer:
    def __init__(self, model, server_port, gloo_group, model_info="aquila-34b") -> None:
        self.model = model
        self.server_port = server_port
        self.model_info = model_info
        self.gloo_group = gloo_group

    def init_flask(self):
        from fastapi import FastAPI, Request

        app = FastAPI()
    
        @app.post("/stream_func")
        async def get_generate_h(request: Request):
            json_post_raw = await request.json()
            config = json.loads(json_post_raw)

            print("request come in")
            print("json_post_raw:", json_post_raw)
            prompts = config["prompt"]
            topk= config.get("top_k_per_token", 20)
            topp = config.get("top_p", 0.9)
            t = config.get("temperature", 0.9)
            seed = config.get("seed", 123)
            sft = config.get("sft", False)
            max_length=config['max_new_tokens']

            history = config.get("history", [])
            template = config.get("template", "aquila-legacy")
            if template not in ["v1", "bair", "aquila-legacy"]:
                template = "aquila-legacy"

            if seed == 0:
                seed = random.randint(0, 429496729)
            print(f"model info is {self.model_info}")

            assert type(prompts) is str
            if sft:
                prompts = covert_prompt_to_input_ids_with_history(prompts, history, tokenizer, 2048, template)
            
            prompts = [prompts,]

            await lock_stream.acquire()
            f = open("disconnected.txt", "w")
            f.close()
            if torch_xmlir:
                torch.distributed.barrier(self.gloo_group)
            else:
                choice = torch.cuda.LongTensor([1])
                torch.distributed.broadcast(choice, 0)
            fun = generate_and_post_process_single_thread(
                                        model,
                                        prompts=prompts,
                                        tokens_to_generate=max_length,
                                        return_output_log_probs=True,
                                        top_k_sampling=topk,
                                        top_p_sampling=topp,
                                        top_p_decay=0.0,
                                        top_p_bound=0.0,
                                        temperature=t,
                                        add_BOS=False,
                                        use_eod_token_for_early_termination=True,
                                        stop_on_double_eol=False,
                                        stop_on_eol=False,
                                        prevent_newline_after_colon=False,
                                        random_seed=seed,
                                        stream=True,
                                        lock_stream=lock_stream)
            if torch_xmlir:
                if param_scope.xacc.eager("false") == "true":
                    torch_xmlir.xpu.empty_cache()
                else:
                    pass
            else:
                torch.cuda.empty_cache()

            def trans():
                while True:
                    try:
                        start_time = time.time()
                        print("start time:", datetime.datetime.now())
                        yield next(fun)
                        print("end time:", datetime.datetime.now())
                        print(f"spend time is {time.time() - start_time}\n")
                    except Exception as e:
                        print(f"e is {e}")
                        #lock_stream.release()

                        break

            def postprocessing(fun):
                f = open("disconnected.txt", "r")
                disconnect_content = f.readlines()
                if 'disconnected' in disconnect_content:
                    print("!!!!!!!!! is disconected")
                    try:
                        next(fun)
                    except Exception as e:
                        print(f"e is {e}")
                print("in postprocessing", flush=True)


            return StreamingResponse(trans(), media_type="text/plain",
                                background=BackgroundTask(postprocessing, fun), lock=lock_stream)


        return app

    def run(self):
        app = self.init_flask()
        uvicorn.run(app, host='0.0.0.0', port=self.server_port, workers=1)

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    config = core_transformer_config_from_args(get_args())

    print_rank_0('building GPT model ...')
    model = GPTModel(config, num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)

    return model

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    return parser


if __name__ == "__main__":

    if torch_xmlir:
        from hyperparameter import param_scope
        ps = param_scope(**{"xacc":{"eager":"true"}})
        ps.__enter__()
        param_scope.frozen()

    import os
    rank = os.getenv("RANK", "0")
    os.makedirs("log_file", exist_ok=True)
    fd = os.open(f"log_file/{rank}.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    
    f = open("disconnected.txt", "w")
    f.close()

    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})
    if torch_xmlir:
        xpu_to_cpu_mapping = {0:0, 1:1, 2:2, 3:3, 4:32, 5:33, 6:34, 7:35}
        os.sched_setaffinity(os.getpid(), [xpu_to_cpu_mapping[torch_xmlir.xpu.current_device()]])

    args = get_args()
    print(f"args is {args}")
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text "
                 "generation.")
    args.exit_on_missing_checkpoint = True
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    print(f"get model successfully....")

    if args.load is not None:
        _ = load_checkpoint(model, None, None)
    print("loading model successfully....")

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    if torch_xmlir:
        # global gloo_group
        gloo_group = torch.distributed.new_group(backend='gloo', timeout=timedelta(days=365))
    else:
        gloo_group = None

    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        server_ins = UvicornServer(model, server_port=int(args.server_port), gloo_group=gloo_group, model_info=args.model_info)
        server_ins.run()

    while True:
        if torch_xmlir:
            torch.distributed.barrier(gloo_group)
            try:
                print("start time:", datetime.datetime.now())
                generate_and_post_process_single_thread(model)
                if torch_xmlir:
                    if param_scope.xacc.eager("false") == "true":
                        torch_xmlir.xpu.empty_cache()
                    else:
                        pass
                else:
                    torch.cuda.empty_cache()
                print("end time:", datetime.datetime.now(), '\n')

            except ValueError as ve:
                pass
        else:
            choice = torch.cuda.LongTensor(1)
            torch.distributed.broadcast(choice, 0)
            if choice[0].item() == 1:
                try:
                    generate_and_post_process_single_thread(model, 
                                            )
                    torch.cuda.empty_cache()

                except Exception as ve:
                    print(f"value error is {ve}")
                    pass
