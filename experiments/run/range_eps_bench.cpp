#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <thread>
#include <atomic>
#include <fcntl.h>
#include <iomanip>

// #include "rmi/books_rmi.h"
#include "FALCON/utils/include.hpp"
#include "FALCON/utils/utils.hpp"
#include "FALCON/Falcon.hpp"
#include "FALCON/cache/CacheInterface.hpp"
#include "./range_bench.cpp"
#include "./config.hpp"

void run_range_eps(RangeEpsConfig& cfg){
    // 1) load data and queries
    std::string file         = falcon::DATASETS + cfg.dataset;
    std::string query_file   = falcon::DATASETS + cfg.dataset + ".range.bin";
    std::vector<KeyType> data    = load_data_pgm_safe<KeyType>(file, cfg.num_keys);
    std::vector<falcon::RangeQ>  all_queries = load_ranges_pgm_safe(query_file);
    size_t MemoryBudget = static_cast<size_t>(cfg.memory_mb) * 1024ull * 1024ull;
    size_t N = all_queries.size();
    size_t eval_begin = static_cast<size_t>(0.3 * N);  // 30% profiling，70% evaluation
    std::vector<falcon::RangeQ> queries(
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
                return bench_range<E>(data, queries, file,
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