#!/bin/bash

set -ex

print_help() {
    echo "Example: $0 <model_path>"
    echo "Args:"
    echo "  model_path: Path to the model, HuggingFace format."
}

if [ $# -ne 1 ]; then
    print_help; exit 1;
fi

python convert_hf_to_gguf.py $1 --outfile $1/ggml_model_f16.gguf
echo "Convert $1 to gguf format ($1/ggml_model_f16.gguf.) successfully."
