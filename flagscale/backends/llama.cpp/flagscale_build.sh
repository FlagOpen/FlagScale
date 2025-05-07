#!/bin/bash

set -e

print_help() {
    echo "Example: $0 <llamma_cpp_backend>"
    echo "Args:"
    echo "  llamma_cpp_backend: Assign backend, now support: cpu, metal, blas, openblas, blis, cuda, gpu, musa, vulkan_mingw64, vulkan_msys2, opencl_android, opencl_windows_arm64"
}

if [ $# -ne 1 ]; then
    print_help; exit 1;
fi

case "$1" in
        cpu|metal|cpu_and_metal)
            cmake -B build
            cmake --build build --config Release
            ;;
        blas|openblas)
            cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
            cmake --build build --config Release
            ;;
        blis)
            # You can skip this step if  in oneapi-basekit docker image, only required for manual installation
            source /opt/intel/oneapi/setvars.sh
            cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_NATIVE=ON
            cmake --build build --config Release
            ;;
        cuda|gpu)
            cmake -B build -DGGML_CUDA=ON
            cmake --build build --config Release
            ;;
        musa)
            cmake -B build -DGGML_MUSA=ON
            cmake --build build --config Release
            ;;
        vulkan_mingw64)
            cmake -B build -DGGML_VULKAN=ON
            cmake --build build --config Release
            ;;
        cann)
            cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release
            cmake --build build --config release
            ;;
        arm_kleidi)
            cmake -B build -DGGML_CPU_KLEIDIAI=ON
            cmake --build build --config Release
            ;;
        hip|vulkan_w64devkit|vulkan_msys2|opencl_android|opencl_windows_arm64)
            echo "auto build unsupport: $1, follow the README.md to build manually:"
            echo "https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md"
            exit 1
            ;;  
        *)
            echo "unknown backend: $1"
            print_help
            exit 1
            ;;
    esac

echo "llama_cpp_backend:" ${llamma_cpp_backend} "build done."
