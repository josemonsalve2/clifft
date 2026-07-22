# CMakeLists.txt for MFMA sweep microbenchmarks
#
# Standalone build (from this directory):
#   mkdir build && cd build
#   cmake .. -DCMAKE_HIP_ARCHITECTURES=gfx942
#   make -j
#
# Or build directly with hipcc:
#   hipcc -O3 --offload-arch=gfx942 -ffp-contract=off \
#         -o mfma_sweep_bench mfma_sweep_bench.hip

cmake_minimum_required(VERSION 3.20)
project(mfma_sweep_bench LANGUAGES CXX HIP)

if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx942)
endif()

set(CMAKE_HIP_STANDARD 17)

add_executable(mfma_sweep_bench mfma_sweep_bench.hip)
set_source_files_properties(mfma_sweep_bench.hip PROPERTIES LANGUAGE HIP)
target_compile_options(mfma_sweep_bench PRIVATE -O3 -ffp-contract=off)

find_package(hip REQUIRED)
target_link_libraries(mfma_sweep_bench PRIVATE hip::host)
