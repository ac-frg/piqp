# This file is part of PIQP.
#
# Copyright (c) 2023 EPFL
#
# This source code is licensed under the BSD 2-Clause License found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.21)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

# Google Benchmark
include(FetchContent)
FetchContent_Declare(
    benchmark
    URL https://github.com/google/benchmark/archive/84c71faa8126e4eedc2bb520352615cb4484d6ad.zip
)
option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
FetchContent_GetProperties(benchmark)
if(NOT benchmark_POPULATED)
    FetchContent_Populate(benchmark)
    add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# copy data
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_library(pipq-benchmark INTERFACE)
target_compile_options(pipq-benchmark INTERFACE ${compiler_flags})
target_link_options(pipq-benchmark INTERFACE ${compiler_flags})
target_link_libraries(pipq-benchmark INTERFACE piqp benchmark::benchmark)

add_executable(chain_mass_sqp_benchmark src/chain_mass_sqp_benchmark.cpp)
target_link_libraries(chain_mass_sqp_benchmark PRIVATE pipq-benchmark Matio::Matio)

add_executable(dense_cholesky_factorization_benchmark src/dense_cholesky_factorization_benchmark.cpp)
target_link_libraries(dense_cholesky_factorization_benchmark PRIVATE pipq-benchmark)

add_executable(dense_sparse_solver_benchmark src/dense_sparse_solver_benchmark.cpp)
target_link_libraries(dense_sparse_solver_benchmark PRIVATE pipq-benchmark)
