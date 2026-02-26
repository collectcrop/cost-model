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
# Benchmark Library
# =========================
add_library(falcon_bench_lib
    # experiments/point_bench.cpp
    experiments/run/epsilon_bench.cpp
    experiments/run/thread_bench.cpp
    experiments/run/batch_bench.cpp
    experiments/run/worker_bench.cpp
    experiments/run/range_bench.cpp
    experiments/run/range_eps_bench.cpp
    experiments/run/range_single_bench.cpp
)

target_include_directories(falcon_bench_lib PUBLIC include)

target_link_libraries(falcon_bench_lib
    PUBLIC falcon_core
)

add_library(bplustree STATIC
    include/bplustree/stx_disk_kv.cpp
)
target_include_directories(bplustree PUBLIC include)

# =========================
# Benchmark Executable
# =========================
add_executable(falcon_bench
    experiments/main.cpp
)

target_link_libraries(falcon_bench
    PRIVATE
        falcon_bench_lib
        CLI11::CLI11
)

add_executable(pgm_disk_test
    experiments/benchmark/pgm_disk_test.cpp
)
target_include_directories(pgm_disk_test PUBLIC include)
target_link_libraries(pgm_disk_test 
    PUBLIC Threads::Threads
    PUBLIC OpenMP::OpenMP_CXX
)

add_executable(aulid_test
    experiments/benchmark/aulid_test.cpp
)
target_include_directories(aulid_test PUBLIC include)
target_link_libraries(aulid_test 
    PUBLIC Threads::Threads
    PUBLIC OpenMP::OpenMP_CXX
)


add_executable(bplustree_test
    experiments/benchmark/bplustree_test.cpp
)
target_include_directories(bplustree_test PUBLIC include)
target_link_libraries(bplustree_test 
    PUBLIC Threads::Threads
    PUBLIC OpenMP::OpenMP_CXX
    PRIVATE bplustree
)

add_executable(page_fetch_test
    experiments/page_fetch_strategy.cpp
)
target_include_directories(page_fetch_test PUBLIC include)
target_link_libraries(page_fetch_test
    PUBLIC Threads::Threads
    PUBLIC OpenMP::OpenMP_CXX
    falcon_core
)