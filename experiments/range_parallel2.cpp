// range_parallel.cpp
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

#include "rmi/books_rmi.h"
#include "utils/include.hpp"
#include "utils/utils.hpp"
#include "utils/LatencyRecorder.hpp"
#include "FALCON/Falcon.hpp"
#include "cache/CacheInterface.hpp"

using KeyType = uint64_t;
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"


struct BenchmarkResult {
    size_t epsilon;
    double avg_latency_ns;   // avg per-query wall time
    double hit_ratio;        // cache hit ratio
    long long wall_ns;       // total wall time (ns)
    long long data_io_ns;    // accumulated storage I/O time (ns) reported by FALCON
    size_t height;           // PGM height
    uint64_t physical_ios;   // merged physical I/Os (as counted by FALCON)
    uint64_t logical_ios;
};

// ---- Worker for range queries ----
template <size_t Epsilon>
static void worker_range(falcon::FalconPGM<uint64_t, Epsilon, 4>* F,
                         const std::vector<pgm::RangeQ>& queries,
                         size_t begin, size_t end,
                         LatencyRecorder* latency) {
    constexpr size_t BATCH = 128;
    using clock_t = std::chrono::high_resolution_clock;
    size_t i = begin;
    std::vector<std::future<falcon::RangeResult>> futs;
    futs.reserve(BATCH);
    auto thread_t0 = clock_t::now();
    while (i < end) {
        futs.clear();
        const size_t j = std::min(end, i + BATCH);
        for (; i < j; ++i) {
            const auto& q = queries[i];
            // 可在此校验 lo<=hi；若文件已保证可省略
            futs.emplace_back(F->range_lookup(q.lo, q.hi));
        }
        for (auto& f : futs) {
            // 结果向量里就是命中的所有 keys；这里不做额外处理
            (void)f.get();
        }
    }
    auto thread_t1 = clock_t::now();
    auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(thread_t1 - thread_t0).count();
    latency->set(begin, dt); 
}

// ---- Benchmark driver ----
template <size_t Epsilon>
static BenchmarkResult bench_range(std::vector<KeyType> data,
                                   std::vector<pgm::RangeQ> ranges,
                                   const std::string& datafile,
                                   pgm::CachePolicy policy,
                                   int num_threads,
                                   size_t memory_budget_bytes,
                                   pgm::IOInterface io_iface = pgm::IO_URING) {
    using Clock = std::chrono::high_resolution_clock;

    // 1) Build PGM (in-memory; for page window estimation only)
    pgm::PGMIndex<KeyType, Epsilon> index(data);  // height/size 可用于统计

    // 2) Open data file with O_DIRECT to bypass OS cache (真实设备性能)
    int fd = ::open(datafile.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open data"); std::exit(1); }

    // 3) Construct FALCON engine (worker 持有缓存与 I/O 接口)
    falcon::FalconPGM<uint64_t, Epsilon, /*EpsRec*/4> F(
        index,
        fd,
        io_iface,
        /*memory_budget_bytes=*/ memory_budget_bytes,
        /*cache_policy=*/ policy,
        /*cache_shards=*/ 1,
        /*max_pages_per_batch=*/ 256,
        /*max_wait_us=*/ 50,
        /*workers=*/ std::min(std::max(num_threads/8, 1), 16)
    );
    const size_t Q = ranges.size();
    LatencyRecorder *latency = new LatencyRecorder(Q);

    // 4) Multi-thread fire
    auto t0 = Clock::now();
    std::vector<std::thread> ths;
    ths.reserve(num_threads);
    size_t per = ranges.size() / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        size_t L = t * per;
        size_t R = (t == num_threads - 1) ? ranges.size() : (t + 1) * per;
        ths.emplace_back(worker_range<Epsilon>, &F, std::cref(ranges), L, R, latency);
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
    out.wall_ns       = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    out.avg_latency_ns= latency->get_avg();
    out.hit_ratio     = hit_ratio;
    out.data_io_ns    = st.io_ns;
    out.height        = index.height();
    out.physical_ios  = st.physical_ios;
    out.logical_ios   = st.logical_ios;
    ::close(fd);
    return out;
}

int main(int argc, char** argv) {
    // 数据与查询文件（按需修改）
    const std::string dataset  = "wiki_ts_200M_uint64_unique";        // fb_10M_uint64_unique
    const std::string datafile = std::string(DATASETS) + dataset;
    const std::string rangefile= std::string(DATASETS) + "wiki_ts_200M_uint64_unique.range.bin";     
    // 基础参数
    const int    N_KEYS     = 200000000;
    const size_t MEM_BUDGET = 10ull * 1024 * 1024; 
    size_t repeat = 3;
    // 读取数据与 range 查询
    auto data   = load_data_pgm_safe<KeyType>(datafile, N_KEYS);
    auto ranges = load_ranges_pgm_safe(rangefile);
    if (data.empty() || ranges.empty()) return 1;

    // 输出 CSV
    std::ofstream csv(dataset+"_falcon_range.csv", std::ios::out | std::ios::trunc);
    csv << "baseline,epsilon,threads,latency_ns,wall_time_s,hit_ratio,avg_IOs,data_io_ns\n";
    csv << std::fixed << std::setprecision(6);

    // 线程与策略
    size_t eps = 16;
    const pgm::CachePolicy policy = pgm::CachePolicy::NONE;

    for (int r = 0; r < repeat; ++r) {
        for (int threads : {1,2,4,8,16,32,64,128,256,512,1024}) {          
            size_t idx_est = 16ull * N_KEYS / (2*eps);
            size_t buf_budget = (MEM_BUDGET > idx_est) ? (MEM_BUDGET - idx_est) : 0;
            BenchmarkResult r;
            bench_range<16>(data, ranges, datafile, policy, threads, buf_budget);

            std::cout << "[T=" << threads << "] eps=" << r.epsilon
                    << " | avg=" << r.avg_latency_ns << " ns"
                    << " | hit=" << r.hit_ratio
                    << " | wall=" << (r.wall_ns / 1e9) << " s"
                    << " | pIOs=" << r.logical_ios
                    << " | io_ns=" << r.data_io_ns
                    << std::endl;

            csv << "FALCON," << r.epsilon << "," << threads << ","
                << r.avg_latency_ns << "," << (r.wall_ns/1e9) << ","
                << r.hit_ratio << "," << r.logical_ios << ","
                << r.data_io_ns << "\n";
            csv.flush();
        }
    }

    csv.close();
    return 0;
}
