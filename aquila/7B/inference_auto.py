
######
# Note that this script is used to serve the checkpoints from the continuous training of the Aquila-7B model,
# which was first trained by BMTrain. So if you want to start from scratch, please remove the
# "--rotary-interleaved-patch" arguments.
######

import argparse
import os 

parser = argparse.ArgumentParser(
        prog='7b-base-server',
    )
parser.add_argument('--server-port', required=True, type=int)
parser.add_argument('--master-process', required=True, type=int)
parser.add_argument('--device', default='0', type=str)
parser.add_argument('--iteration', required=False, type=int, default=-1)
parser.add_argument('--checkpoint-path', required=True, type=str)
parser.add_argument('--model-info', required=True, type=str)

args = parser.parse_args()

model_iteration = args.iteration
server_port = args.server_port
master_process = args.master_process
device_number = args.device
checkpoint_path = args.checkpoint_path
model_info = args.model_info

if model_iteration != -1:
    with open(os.path.join(checkpoint_path, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write(str(model_iteration))

sh_content = """#!/bin/bash

DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port {master_port}"

CHECKPOINT={checkpoint_path}
VOCAB_FILE=../aquila/tokenizer/vocab.json
MERGE_FILE=../aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=../aquila/tokenizer/special_tokens.txt

export CUDA_DEVICE_MAX_CONNECTIONS=1

CUDA_VISIBLE_DEVICES={device_number} torchrun $DISTRIBUTED_ARGS tools/run_text_generation_uvicorn_server.py \
       --server-port {server_port} \
       --model-info {model_info} \
       --use-flash-attn \
       --rotary-interleaved-patch \
       --apply-layernorm-rms \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --num-layers 32  \
       --hidden-size 4096  \
       --load ${CHECKPOINT}  \
       --disable-bias-linear \
       --layernorm-epsilon 1e-5 \
       --num-attention-heads 32  \
       --max-position-embeddings 2048  \
       --use-rotary-position-embeddings \
       --no-position-embedding \
       --swiglu \
       --multiple-of 256 \
       --untie-embeddings-and-output-weights \
       --tokenizer-type AquilaTokenizer  \
       --fp16  \
       --micro-batch-size 1  \
       --seq-length 2048  \
       --out-seq-length 2048  \
       --temperature 1.0  \
       --vocab-file $VOCAB_FILE  \
       --merge-file $MERGE_FILE  \
       --top_p 0.9  \
       --seed 42   \
       --special-tokens-file $SPECIAL_TOKENS_FILE 
       
"""

sh_dir = "./examples/aquila/7B/server"
os.makedirs(sh_dir, exist_ok=True)

sh_filename = os.path.join(sh_dir, f"{model_info}.sh")

print(f"server filename is {sh_filename}")

with open(sh_filename, "w") as f:
    sh_content = sh_content.replace("{master_port}", str(master_process))
    sh_content = sh_content.replace("{checkpoint_path}", checkpoint_path)
    sh_content = sh_content.replace("{device_number}", "'" + device_number + "'")
    sh_content = sh_content.replace("{server_port}", str(server_port))
    sh_content = sh_content.replace("{model_info}", model_info)
    
    f.write((sh_content))

import subprocess
import signal

run_cmd = f"sh {sh_filename}"

p = subprocess.Popen(run_cmd, shell=True, preexec_fn=os.setsid)
def signal_handler(signal, frame):
    os.killpg(os.getpgid(p.pid), 9)
signal.signal(signal.SIGINT, signal_handler)
p.wait()
print ('finish')