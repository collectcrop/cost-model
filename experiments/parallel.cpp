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
#include <thread>
#include <mutex>
#include <atomic>
#include <fcntl.h>
#include "distribution/zipf.hpp"
#include "pgm/pgm_index.hpp"
#include "utils/include.hpp"
#include "utils/utils.hpp"
#include "FALCON/Falcon.hpp"      // ← 新增：FALCON 入口
#include "cache/CacheInterface.hpp"  // 如果你的 MakeShardedCache 在这里声明

using KeyType = uint64_t;
#define DIRECTORY "/mnt/home/zwshi/learned-index/cost-model/experiments/"
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"

struct Record {
    uint64_t key;
};

struct BenchmarkResult {
    size_t epsilon;
    double time_ns;
    double hit_ratio;
    time_t total_time;
    time_t data_IO_time;
    size_t height;
    size_t data_IOs;
};


using timer = std::chrono::high_resolution_clock;

// 全局统计
std::atomic<size_t> global_queries_done{0};

// 单线程执行函数
template <size_t Epsilon, size_t M>
void worker_thread(falcon::FalconPGM<uint64_t, Epsilon, 4>* F,      // ← 用 FALCON
                   const std::vector<KeyType>& queries,
                   size_t begin, size_t end,
                   std::atomic<long long>& total_time_ns) {
    constexpr size_t BATCH = 128;  
    auto t0 = timer::now();

    std::vector<std::future<falcon::PointResult>> futs;
    futs.reserve(BATCH);

    size_t i = begin;
    while (i < end) {
        futs.clear();
        size_t j = std::min(end, i + BATCH);
        for (; i < j; ++i) {
            futs.emplace_back(F->point_lookup(queries[i]));
        }
        for (auto& f : futs) {
            (void)f.get(); // 可选：统计 found 数；这里只计时
            global_queries_done.fetch_add(1, std::memory_order_relaxed);
        }
    }

    auto t1 = timer::now();
    auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    total_time_ns.fetch_add(dt, std::memory_order_relaxed);
}


// 多线程 benchmark
template <size_t Epsilon, size_t M>
BenchmarkResult benchmark_mt(std::vector<KeyType> data,
                             std::vector<KeyType> queries,
                             std::string filename,
                             pgm::CacheStrategy s,
                             int num_threads) {
    // 1) 只构建 PGM（用于页窗口估计；不让它自己做 I/O/缓存）
    pgm::PGMIndex<KeyType, Epsilon> index(data);

    // 2) 打开数据文件（FALCON 持有 fd）
    int data_fd = ::open(filename.c_str(), O_RDONLY | O_DIRECT);
    if (data_fd < 0) { perror("open data"); std::exit(1); }

    // 3) 策略映射
    pgm::CachePolicy policy = pgm::CachePolicy::LRU;
    switch (s) {
        case pgm::CacheStrategy::LRU:  policy = pgm::CachePolicy::LRU;  break;
        case pgm::CacheStrategy::FIFO: policy = pgm::CachePolicy::FIFO; break;
        case pgm::CacheStrategy::LFU:  policy = pgm::CachePolicy::LFU;  break;
    }

    // 4) 构建 FALCON 引擎
    //    - MemoryBudget 用运行时参数 M（字节）
    //    - workers = num_threads（也可先设 1，简化确定性；两种都可）
    falcon::FalconPGM<uint64_t, Epsilon, /*EpsRec*/ 4> F(
        index,
        data_fd,
        pgm::IO_URING,
        /*memory_budget_bytes=*/ M,
        /*cache_policy=*/ policy,
        /*cache_shards=*/ 0,                
        /*max_pages_per_batch=*/ 256,       // 你也可试 64/128/512
        /*max_wait_us=*/ 50,                // 时间窗口；如果想确定性，可把 worker 里做“计数触发”
        /*workers=*/ std::max(num_threads/4, 1)   
    );

    // 5) 多线程提交查询（每线程用批量 futures）
    std::atomic<long long> total_time_ns{0};
    global_queries_done = 0;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    size_t per_thread = queries.size() / num_threads;

    auto start_all = timer::now();
    for (int t = 0; t < num_threads; t++) {
        size_t begin = t * per_thread;
        size_t end   = (t == num_threads - 1) ? queries.size() : (t + 1) * per_thread;
        threads.emplace_back(worker_thread<Epsilon, M>, &F, std::ref(queries), begin, end, std::ref(total_time_ns));
    }
    for (auto& th : threads) th.join();
    auto end_all = timer::now();
    auto wall_clock_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_all - start_all).count();

    // 6) 从 FALCON 拉统计
    auto st = F.stats();
    double hit_ratio = 0.0;
    auto hm = st.cache_hits + st.cache_misses;
    if (hm) hit_ratio = double(st.cache_hits) / double(hm);

    BenchmarkResult result;
    result.epsilon       = Epsilon;
    result.time_ns       = double(wall_clock_ns) / queries.size();  // 平均 query latency（墙钟）
    result.hit_ratio     = hit_ratio;
    result.total_time    = wall_clock_ns;    // ns
    result.data_IO_time  = st.io_ns;         // FALCON 收集的 I/O 时间累计（ns）
    result.height        = index.height();   // PGM 高度
    result.data_IOs      = st.physical_ios;  // 物理 I/O 片段数（合并后）
    ::close(data_fd);
    return result;
}

int main() {
    std::string dataset = "books";
    std::string filename = "books_10M_uint64_unique";
    std::string query_filename = "books_10M_uint64_unique.query.bin";
    std::string file = DATASETS + filename;         
    std::string query_file = DATASETS + query_filename;

    std::vector<KeyType> data    = load_data(file, 10000000);
    std::vector<KeyType> queries = load_queries(query_file);

    // const size_t MemoryBudget = 20ull * 1024 * 1024;
    const size_t MemoryBudget = 0;

    for (int e = 0; e <= 14; ++e) {
        uint64_t threads = 1ULL << e;
        auto result = benchmark_mt<8, MemoryBudget>(data, queries, file, pgm::CacheStrategy::LRU, threads);

        std::cout << "[Threads=" << threads << "] ε=" << result.epsilon
                  << ", avg query time=" << result.time_ns << " ns"
                  << ", hit ratio=" << result.hit_ratio
                  << ", total wall time=" << result.total_time / 1e9 << " s"
                  << ", data IOs=" << result.data_IOs << std::endl;
    }
    return 0;
}
