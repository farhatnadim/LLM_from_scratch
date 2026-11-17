# LibTorch as an interface target, made from the torch/ folder of the PyTorch
# package in .venv (docs/specification.md, S-1).
#
# This file does not call find_package(Torch). On the machine of the course,
# find_package(Torch) fails in three ways:
#   1. The shell exports TORCH_ROOT, which points to an old PyTorch.
#   2. A CUDA build of LibTorch starts the CUDA compiler, which refuses
#      GCC 15 and Clang 21.
#   3. A global CMAKE_CXX_STANDARD goes into that CUDA probe.
# The PyTorch package brings its own CUDA libraries. No CUDA toolkit is necessary.

set(LIBTORCH_DIR "" CACHE PATH "The torch/ folder of the PyTorch package (CMakePresets.json sets it)")

if(NOT EXISTS "${LIBTORCH_DIR}/include/torch/csrc/api/include/torch/torch.h")
    message(FATAL_ERROR
        "LibTorch not found in LIBTORCH_DIR='${LIBTORCH_DIR}'.\n"
        "Run 'uv sync' in the repository root, then configure with a preset: 'cmake --preset clang'.")
endif()

add_library(libtorch INTERFACE)
target_include_directories(libtorch SYSTEM INTERFACE
    "${LIBTORCH_DIR}/include"
    "${LIBTORCH_DIR}/include/torch/csrc/api/include")
target_link_directories(libtorch INTERFACE "${LIBTORCH_DIR}/lib")
target_link_libraries(libtorch INTERFACE torch torch_cpu c10)

set(LIBTORCH_HAS_CUDA OFF)
if(EXISTS "${LIBTORCH_DIR}/lib/libtorch_cuda.so")
    set(LIBTORCH_HAS_CUDA ON)
    # No code calls torch_cuda directly. --no-as-needed keeps the link, so that
    # the CUDA device is visible at run time.
    target_link_libraries(libtorch INTERFACE -Wl,--no-as-needed torch_cuda c10_cuda -Wl,--as-needed)
endif()
target_link_options(libtorch INTERFACE "-Wl,-rpath,${LIBTORCH_DIR}/lib")

message(STATUS "LibTorch: ${LIBTORCH_DIR} (CUDA: ${LIBTORCH_HAS_CUDA})")
