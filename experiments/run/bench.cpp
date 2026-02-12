#ifndef BENCH_H
#define BENCH_H
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
#include <iostream>
#include <fcntl.h>
#include "distribution/zipf.hpp"
#include "FALCON/pgm/pgm_index.hpp"
#include "FALCON/utils/include.hpp"
#include "FALCON/utils/utils.hpp"
#include "FALCON/utils/LatencyRecorder.hpp"
#include "FALCON/Falcon.hpp"     
#include "FALCON/cache/CacheInterface.hpp" 
#include "./config.hpp"

using KeyType = uint64_t;
using timer = std::chrono::high_resolution_clock;

template <size_t Epsilon>
void worker_thread(falcon::FalconPGM<uint64_t, Epsilon, 4>* F,
                   const std::vector<KeyType>& queries,
                   size_t begin, size_t end,
                   std::atomic<long long>& total_time_ns,
                   LatencyRecorder* latency,
                   size_t batch_size) {
    using clock_t = std::chrono::high_resolution_clock;

    struct Pending {
        size_t qid;
        clock_t::time_point t0;
        std::future<falcon::PointResult> fut;
    };

    auto thread_t0 = clock_t::now();

    std::vector<Pending> inflight;
    inflight.reserve(batch_size);

    size_t i = begin;
    while (i < end) {
        inflight.clear();
        const size_t j = std::min(end, i + batch_size);

        // submission phase
        for (; i < j; ++i) {
            auto t0 = clock_t::now();                      
            auto fut = F->point_lookup(queries[i]);       
            inflight.push_back(Pending{ i, t0, std::move(fut) });
        }

        // completion phase
        for (auto& p : inflight) {
            (void)p.fut.get();                            // wait for completion
            auto t1 = clock_t::now();
            uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - p.t0).count();
            latency->set(p.qid, ns);                       
            global_queries_done.fetch_add(1, std::memory_order_relaxed);
        }
    }

    auto thread_t1 = clock_t::now();
    auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(thread_t1 - thread_t0).count();
    // latency->set(begin, dt); 
    total_time_ns.fetch_add(dt, std::memory_order_relaxed);
}


template <size_t Epsilon>
BenchmarkResult benchmark_mt(std::vector<KeyType> data,
                             std::vector<KeyType> queries,
                             const std::string &filename,
                             falcon::CachePolicy s,
                             int num_threads,
                             size_t memory_budget_bytes,
                             size_t batch_size = 128,
                             int worker_threads = -1) {
    // 1) construct PGM
    pgm::PGMIndex<KeyType, Epsilon> index(data);
    // 2) open data file
    int data_fd = ::open(filename.c_str(), O_RDONLY|O_DIRECT);
    if (data_fd < 0) {
        perror("open data");
        std::exit(1);
    }

    // 3) strategy cast
    falcon::CachePolicy policy = falcon::CachePolicy::NONE;
    switch (s) {
        case falcon::CachePolicy::LRU:  policy = falcon::CachePolicy::LRU;  break;
        case falcon::CachePolicy::FIFO: policy = falcon::CachePolicy::FIFO; break;
        case falcon::CachePolicy::LFU:  policy = falcon::CachePolicy::LFU;  break;
        case falcon::CachePolicy::NONE: policy = falcon::CachePolicy::NONE; break;
    }

    // 4) init FALCON
    falcon::FalconPGM<uint64_t, Epsilon, /*EpsRec*/ 4> F(
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

    // 5) run queries with multiple threads
    std::atomic<long long> total_time_ns{0};
    global_queries_done = 0;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    size_t per_thread = queries.size() / num_threads;

    const size_t Q = queries.size();
    LatencyRecorder *latency = new LatencyRecorder(Q);

    auto start_all = timer::now();
    for (int t = 0; t < num_threads; t++) {
        size_t begin = t * per_thread;
        size_t end   = (t == num_threads - 1) ? queries.size() : (t + 1) * per_thread;
        threads.emplace_back(worker_thread<Epsilon>, &F, std::ref(queries),
                             begin, end, std::ref(total_time_ns), latency, batch_size);
    }
    for (auto &th : threads) th.join();
    auto end_all = timer::now();
    auto wall_clock_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_all - start_all).count();

    // 6) stats
    auto st = F.stats();
    double hit_ratio = 0.0;
    auto hm = st.cache_hits + st.cache_misses;
    if (hm) hit_ratio = double(st.cache_hits) / double(hm);

    BenchmarkResult result;
    result.epsilon      = Epsilon;
    result.avg_lat      = latency->get_avg();
    result.p50_lat      = latency->p50();
    result.p95_lat      = latency->p95();
    result.p99_lat      = latency->p99();
    result.hit_ratio    = hit_ratio;
    result.total_time   = wall_clock_ns;
    result.height       = index.height();
    result.data_IOs     = st.logical_ios;

    ::close(data_fd);
    delete latency;
    return result;
}

#endif