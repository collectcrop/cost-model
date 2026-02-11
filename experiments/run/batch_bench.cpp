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
#include <atomic>
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

// struct BenchmarkResult {
//     size_t epsilon;
//     double time_ns;
//     double p50_ns;
//     double p95_ns;
//     double p99_ns;
//     double hit_ratio;
//     time_t total_time;
//     time_t data_IO_time;
//     size_t height;
//     size_t data_IOs;
// };


using timer = std::chrono::high_resolution_clock;

void run_batch(BatchConfig& cfg){
    std::string file = falcon::DATASETS + cfg.dataset;
    std::string query_file = falcon::DATASETS + cfg.dataset + ".query.bin";

    std::vector<KeyType> data    = load_data_pgm_safe<KeyType>(file, cfg.num_keys);
    std::vector<KeyType> queries = load_queries_pgm_safe<KeyType>(query_file);

    for (int batch : cfg.batch_sizes) {
        auto result = benchmark_mt<16>(data, queries, file,
                                       cfg.policy,
                                       1,
                                       static_cast<size_t>(cfg.memory_mb) * 1024ull * 1024ull,
                                       batch);
        std::cout << "[Batch=" << batch << "] ε=" << result.epsilon
                  << ", avg query time=" << result.avg_lat << " ns"
                  << ", hit ratio=" << result.hit_ratio
                  << ", total wall time=" << result.total_time / 1e9 << " s"
                  << ", data IOs=" << result.data_IOs
                  << ", p50=" << result.p50_lat << " ns"
                  << ", p95=" << result.p95_lat << " ns"
                  << ", p99=" << result.p99_lat << " ns"
                  << std::endl;
    }
}
// int main() {
//     std::string dataset = "books";
//     std::string filename = "books_10M_uint64_unique";
//     std::string query_filename = "books_10M_uint64_unique.query.bin";
//     std::string file = falcon::DATASETS + filename;         
//     std::string query_file = falcon::DATASETS + query_filename;
//     int n = 10000000;
//     std::vector<KeyType> data    = load_data(file, n);
//     std::vector<KeyType> queries = load_queries(query_file);

//     const size_t MemoryBudget = 50ull * 1024 * 1024;
//     // const size_t MemoryBudget = 0;

//     std::ofstream csv("0M_batch_falcon.csv", std::ios::out | std::ios::trunc);
//     if (!csv) {
//         std::cerr << "Failed to open CSV output\n";
//         return 1;
//     }
//     csv << "batch,avg_latency_ns,p50_ns,p95_ns,p99_ns,avg_walltime_s,avg_IOs,data_IO_time,hit_ratio\n";
//     csv << std::fixed << std::setprecision(6);
//     uint64_t threads = 16;
//     size_t epsilon = 16;
//     size_t repeats = 3;
//     pgm::CachePolicy s = pgm::CachePolicy::NONE;
//     for (size_t i = 0; i < repeats; i++ ){
//         for (size_t batch : {1,2,4,8,16,32,64,128,256,512,1024,2048,4096}) {     //2,4,6,8,10,12,14,16,18,20,24,32,48,64,128
//             BenchmarkResult result;
//             const size_t M = MemoryBudget - 16*n/(2*epsilon);
            
//             result = benchmark_mt<16>(data, queries, file, s, threads, M, batch);
            
//             std::cout << "[Batch=" << batch << "] ε=" << result.epsilon
//                     << ", avg query time=" << result.time_ns << " ns"
//                     << ", hit ratio=" << result.hit_ratio
//                     << ", total wall time=" << result.total_time / 1e9 << " s"
//                     << ", data IOs=" << result.data_IOs 
//                     << ", data IO time=" << result.data_IO_time / 1e9 << " s" << std::endl;
//             csv << batch << "," << result.time_ns << "," <<  result.p50_ns << ","
//                 << result.p95_ns << ","
//                 << result.p99_ns << "," 
//                 << result.total_time / 1e9 << "," << result.data_IOs << ","
//                 << result.data_IO_time << "," << result.hit_ratio << "\n";
//             csv.flush();
//         }
//     }
//     csv.close();
//     return 0;
// }
