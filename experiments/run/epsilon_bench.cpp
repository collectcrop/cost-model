// test epsilon
#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <thread>
#include <mutex>
#include <array>
#include <fcntl.h>
#include "distribution/zipf.hpp"
#include "FALCON/pgm/pgm_index.hpp"
#include "FALCON/utils/include.hpp"
#include "FALCON/utils/utils.hpp"
#include "FALCON/utils/config.hpp"
#include "FALCON/utils/LatencyRecorder.hpp"
#include "FALCON/Falcon.hpp"     
#include "FALCON/cache/CacheInterface.hpp" 
#include "./bench.cpp"

using KeyType = uint64_t;
using timer = std::chrono::high_resolution_clock;

constexpr std::array<size_t, 15> SupportedEps = {
    2,4,6,8,10,12,14,16,18,20,24,32,48,64,128
};

template<size_t... Es, typename Fn>
BenchmarkResult dispatch_epsilon(size_t eps,
                                 std::index_sequence<Es...>,
                                 Fn&& fn) {

    BenchmarkResult result{};
    bool matched = false;

    ((eps == SupportedEps[Es] && 
    (result = fn.template operator()<SupportedEps[Es]>(), matched = true)), ...);

    if(!matched)
        throw std::runtime_error("Unsupported epsilon");

    return result;
}

void run_epsilon(const EpsilonConfig& cfg) {
    // 1) load data and queries
    std::string file         = falcon::DATASETS + cfg.dataset;
    std::string query_file   = falcon::DATASETS + cfg.dataset + ".query.bin";
    std::vector<KeyType> data    = load_data_pgm_safe<KeyType>(file, cfg.num_keys);
    std::vector<KeyType> all_queries = load_queries_pgm_safe<KeyType>(query_file);
    size_t MemoryBudget = static_cast<size_t>(cfg.memory_mb) * 1024ull * 1024ull;
    size_t N = all_queries.size();
    size_t eval_begin = static_cast<size_t>(0.3 * N);  // 30% profiling，70% evaluation
    std::vector<KeyType> queries(
        all_queries.begin() + eval_begin,
        all_queries.end()
    );
    // 2) run benchmarks for each epsilon
    for (size_t epsilon : cfg.epsilons) {
        if (MemoryBudget < 16*cfg.num_keys/(2*epsilon)){
            std::cout << "Memory budget too small for ε=" << epsilon << ", skipping.\n";
            continue;
        }
        const size_t M = static_cast<size_t>(cfg.memory_mb) * 1024ull * 1024ull - 16*cfg.num_keys/(2*epsilon);
        auto result = dispatch_epsilon(
            epsilon,
            std::make_index_sequence<SupportedEps.size()>(),
            [&]<size_t E>() {
                return benchmark_mt<E>(data, queries, file,
                                    cfg.policy, 1, M);
            }
        );
        std::cout << "[Threads=1] ε=" << epsilon << ", M=" << M  
                  << ", avg latency=" << result.avg_lat << " ns"
                  << ", hit ratio=" << result.hit_ratio
                  << ", total wall time=" << result.total_time / 1e9 << " s"
                  << ", data IOs=" << result.data_IOs
                  << std::endl;
    }
}