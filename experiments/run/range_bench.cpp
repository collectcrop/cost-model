#ifndef BENCH_RANGE_H
#define BENCH_RANGE_H
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
#include "./config.hpp"

using KeyType = uint64_t;

// ---- Worker for range queries ----
template <size_t Epsilon>
static void worker_range(falcon::FalconPGM<uint64_t, Epsilon, 4>* F,
                         const std::vector<falcon::RangeQ>& queries,
                         size_t begin, size_t end) {
    constexpr size_t BATCH = 128;
    size_t i = begin;
    std::vector<std::future<falcon::RangeResult>> futs;
    futs.reserve(BATCH);

    while (i < end) {
        futs.clear();
        const size_t j = std::min(end, i + BATCH);
        for (; i < j; ++i) {
            const auto& q = queries[i];
            futs.emplace_back(F->range_lookup(q.lo, q.hi));
        }
        for (auto& f : futs) {
            (void)f.get();
        }
    }
}

// ---- Benchmark driver ----
template <size_t Epsilon>
static BenchmarkResult bench_range(std::vector<KeyType> data,
                                   std::vector<falcon::RangeQ> ranges,
                                   const std::string& datafile,
                                   falcon::CachePolicy policy,
                                   int num_threads,
                                   size_t memory_budget_bytes,
                                   falcon::IOInterfaceType io_iface = falcon::IO_URING) {
    using Clock = std::chrono::high_resolution_clock;

    // 1) Build PGM (in-memory; for page window estimation only)
    pgm::PGMIndex<KeyType, Epsilon> index(data); 

    // 2) Open data file with O_DIRECT to bypass OS cache
    int fd = ::open(datafile.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open data"); std::exit(1); }

    // 3) Construct FALCON engine
    falcon::FalconPGM<uint64_t, Epsilon, /*EpsRec*/4> F(
        index,
        fd,
        io_iface,
        /*memory_budget_bytes=*/ memory_budget_bytes,
        /*cache_policy=*/ policy,
        /*cache_shards=*/ 1,
        /*max_pages_per_batch=*/ 256,
        /*max_wait_us=*/ 50,
        /*workers=*/ std::max(num_threads/16, 1) 
    );

    // 4) Multi-thread fire
    auto t0 = Clock::now();
    std::vector<std::thread> ths;
    ths.reserve(num_threads);
    size_t per = ranges.size() / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        size_t L = t * per;
        size_t R = (t == num_threads - 1) ? ranges.size() : (t + 1) * per;
        ths.emplace_back(worker_range<Epsilon>, &F, std::cref(ranges), L, R);
    }
    for (auto& th : ths) th.join();
    auto t1 = Clock::now();

    // 5) Stats
    auto st = F.stats();
    double hit_ratio = 0.0;
    const auto hm = st.cache_hits + st.cache_misses;
    if (hm) hit_ratio = double(st.cache_hits) / double(hm);

    BenchmarkResult out;
    out.epsilon       = Epsilon;
    out.total_time    = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    out.avg_lat       = double(out.total_time) / std::max<size_t>(1, ranges.size());
    out.hit_ratio     = hit_ratio;
    out.height        = index.height();
    out.data_IOs      = st.logical_ios;
    ::close(fd);
    return out;
}

#endif