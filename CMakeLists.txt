cmake_minimum_required(VERSION 3.28)

project(FALCON VERSION 1.0 LANGUAGES CXX)
# =========================
# C++ Standard
# =========================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =========================
# Build Type Default
# =========================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# =========================
# Compiler Flags
# =========================
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -DNDEBUG)
endif()

# ================
# Dependencies
# ================
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# =========================
# Fetch CLI11
# =========================
include(FetchContent)

FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.3.2
)

FetchContent_MakeAvailable(CLI11)

# ================
# core libraries
# ================
add_library(falcon_core
    include/FALCON/cache/CacheFactory.cpp
)
target_include_directories(falcon_core PUBLIC include)
target_link_libraries(falcon_core
    PUBLIC Threads::Threads
    PUBLIC OpenMP::OpenMP_CXX
    PUBLIC uring
    PUBLIC aio
)

# =========================
# Benchmark Logic Library
# =========================
add_library(falcon_bench_lib
    # experiments/point_bench.cpp
    experiments/run/epsilon_bench.cpp
    experiments/run/thread_bench.cpp
    experiments/run/batch_bench.cpp
)

target_include_directories(falcon_bench_lib PUBLIC include)

target_link_libraries(falcon_bench_lib
    PUBLIC falcon_core
)

# =========================
# Unified Benchmark Executable
# =========================
add_executable(falcon_bench
    experiments/main.cpp
)

target_link_libraries(falcon_bench
    PRIVATE
        falcon_bench_lib
        CLI11::CLI11
)


# ================
# FALCON benchmarks
# ================

# add_executable(falcon_thread_test experiments/parallel.cpp)
# target_link_libraries(falcon_thread_test PRIVATE falcon_core)

# add_executable(falcon_eps_test experiments/parallel2.cpp)
# target_link_libraries(falcon_eps_test PRIVATE falcon_core)

# add_executable(falcon_batch_test experiments/parallel4.cpp)
# target_link_libraries(falcon_batch_test PRIVATE falcon_core)

# add_executable(falcon_worker_test experiments/parallel6.cpp)
# target_link_libraries(falcon_worker_test PRIVATE falcon_core)

# add_executable(falcon_range_eps_test experiments/range_parallel.cpp)
# target_link_libraries(falcon_range_eps_test PRIVATE falcon_core)

# add_executable(falcon_range_single_test experiments/falcon_range_single.cpp)
# target_link_libraries(falcon_range_single_test PRIVATE falcon_core)

# add_executable(falcon_point_test experiments/falcon_point.cpp)
# target_link_libraries(falcon_point_test PRIVATE falcon_core)

# add_executable(falcon_join_test experiments/join_parallel.cpp)
# target_link_libraries(falcon_join_test PRIVATE falcon_core)

# ================
# other benchmark
# ================

# add_executable(pgm_disk_test experiments/benchmark/pgm_disk_test.cpp)
# target_link_libraries(pgm_disk_test PRIVATE falcon_core)

# add_executable(aulid_test experiments/benchmark/aulid_test.cpp)
# target_link_libraries(aulid_test PRIVATE falcon_core)

# add_executable(cache_policy_test experiments/cache_policy_bench.cpp)
# target_link_libraries(cache_policy_test PRIVATE falcon_core)

# add_executable(page_fetch_test experiments/page_fetch_strategy.cpp)
# target_link_libraries(page_fetch_test PRIVATE falcon_core)

# add_executable(index_build_test experiments/build.cpp)
# target_link_libraries(index_build_test PRIVATE falcon_core)