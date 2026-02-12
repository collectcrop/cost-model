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

template <size_t Epsilon>
BenchmarkResult bench_range_sig(std::vector<KeyType> data,
                                std::vector<KeyType> queries,
                                const std::string &filename,
                                falcon::CachePolicy policy,
                                int num_threads,
                                size_t memory_budget_bytes,
                                size_t batch_size = 128,
                                int worker_threads = -1) {
    pgm::PGMIndex<KeyType, Epsilon> index(data);
    int data_fd = ::open(filename.c_str(), O_RDONLY|O_DIRECT);
    if (data_fd < 0) {
        perror("open data");
        std::exit(1);
    }

    // 4) init FALCON
    falcon::FalconPGM<uint64_t, Epsilon, /*EpsRec*/ 4> engine(
        index,
        data_fd,
        falcon::IO_URING,
        /*memory_budget_bytes=*/ memory_budget_bytes,
        /*cache_policy=*/ policy,
        /*cache_shards=*/ 1,
        /*max_pages_per_batch=*/ 256,
        /*max_wait_us=*/ 50,
        /*workers=*/ (worker_threads==-1)?std::max(num_threads/16, 1):worker_threads 
    );

    uint64_t matched_total = 0;

    auto t0 = timer::now();
    sort(queries.begin(), queries.end());
    auto lo = queries.front(), hi = queries.back();
    auto fut = engine.range_lookup(lo, hi);
    auto rr  = fut.get();
    

    if (!queries.empty()) {
        const auto& Q = queries;
        size_t qL = 0, qR = Q.size();
        // const auto& R = rr.keys;           
        // size_t i = 0, j = 0;
        // while (i < Q.size() && j < R.size()) {
        //     if (Q[i] < R[j]) {
        //         ++i;
        //     } else if (R[j] < Q[i]) {
        //         ++j;
        //     } else {
        //         ++matched_total;
        //         ++i;
        //         ++j;
        //     }
        // }
        for (size_t i = qL; i < qR; ++i) {
            if (std::binary_search(rr.keys.begin(), rr.keys.end(), Q[i])) {
                ++matched_total;
            }
        }
    } else {
        matched_total = rr.keys.size();
    }
    auto t1 = timer::now();

    auto st = engine.stats();
    const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    BenchmarkResult r;
    r.epsilon        = Epsilon;
    r.total_time     = total_ns;
    r.hit_ratio      = (st.cache_hits + st.cache_misses)
                        ? double(st.cache_hits) / double(st.cache_hits + st.cache_misses) : 0.0;
    r.height         = index.height();
    r.data_IOs       = st.logical_ios;
    ::close(data_fd);
    return r;
}

void run_range_sig(RangeSigConfig& cfg){
    std::string file         = falcon::DATASETS + cfg.dataset;
    std::string query_file   = falcon::DATASETS + cfg.query_file;
    std::vector<KeyType> data    = load_data_pgm_safe<KeyType>(file, cfg.num_keys);
    std::vector<KeyType> queries = load_queries_pgm_safe<KeyType>(query_file);
    size_t M = static_cast<size_t>(cfg.memory_mb) * 1024ull * 1024ull;
    
    auto result = bench_range_sig<16>(data, queries, file, cfg.policy, 1, M);
    std::cout << "[Threads=1] ε=" << 16 << ", M=" << M  
                << ", avg latency=" << result.avg_lat << " ns"
                << ", hit ratio=" << result.hit_ratio
                << ", total wall time=" << result.total_time / 1e9 << " s"
                << ", data IOs=" << result.data_IOs
                << std::endl;
    
}