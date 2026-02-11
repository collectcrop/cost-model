// test threads
#include <sched.h>
#include <cstdlib>
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
#include <fcntl.h>
#include "distribution/zipf.hpp"
#include "FALCON/pgm/pgm_index.hpp"
#include "FALCON/utils/include.hpp"
#include "FALCON/utils/utils.hpp"
#include "FALCON/utils/LatencyRecorder.hpp"
#include "FALCON/Falcon.hpp"     
#include "FALCON/cache/CacheInterface.hpp" 
#include "FALCON/utils/config.hpp"
#include "./bench.cpp"

using KeyType = uint64_t;
using timer = std::chrono::high_resolution_clock;

void run_threads(const ThreadConfig& cfg) {
    std::string file = falcon::DATASETS + cfg.dataset;
    std::string query_file = falcon::DATASETS + cfg.dataset + ".query.bin";

    std::vector<KeyType> data    = load_data_pgm_safe<KeyType>(file, cfg.num_keys);
    std::vector<KeyType> queries = load_queries_pgm_safe<KeyType>(query_file);

    for (int threads : cfg.thread_counts) {
        auto result = benchmark_mt<16>(data, queries, file,
                                       cfg.policy,
                                       threads,
                                       static_cast<size_t>(cfg.memory_mb) * 1024ull * 1024ull);

        std::cout << "[Threads=" << threads << "] ε=" << result.epsilon
                  << ", avg query time=" << result.avg_lat << " ns"
                  << ", hit ratio=" << result.hit_ratio
                  << ", total wall time=" << result.total_time / 1e9 << " s"
                  << ", data IOs=" << result.data_IOs
                  << std::endl;
    }
}