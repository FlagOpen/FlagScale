#!/bin/bash

set -ex

print_help() {
    echo "Example: $0 --llamma_cpp_backend cpu --model_path /tmp/models/Qwen3-0.6B"
    echo "Args:"
    echo "  --llamma_cpp_backend   Assign backend (cpu, gpu, metal), cpu is the same with metal when build, default is cpu"
    echo "  --model_path           Path to the model, HuggingFace or gguf format, default is /tmp/models/Qwen3-0.6B"
}

llamma_cpp_backend="cpu"
model_path="/tmp/models/Qwen3-0.6B"

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            print_help
            exit 0
            ;;
        --llamma_cpp_backend)
            llamma_cpp_backend="$2"
            shift 2
            ;;
        --model_path)
            model_path="$2"
            shift 2
            ;;
        *)
            echo "unknown arg: $1"
            shift
            print_help
            exit 1
            ;;
    esac
done

echo $0 "Args:"
echo "------------------------"
echo "llamma_cpp_backend:" ${llamma_cpp_backend}
echo "model_path        :" ${model_path}
echo "------------------------"

# CLEAN
rm -rf ./build
rm -rf ${model_path}/ggml_model_f16.gguf
# BUILD
./flagscale_build.sh ${llamma_cpp_backend} 
# CONVERT
./flagscale_convert.sh ${model_path}
