cmake_minimum_required(VERSION 3.18)
project(cuda_matrix_multiply LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul src/matmul.cu)

# Set a reasonable default architecture; override with
#   cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES "52;61;75")
endif()
