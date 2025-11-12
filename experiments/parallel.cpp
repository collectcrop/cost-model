// test threads
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
#include "pgm/pgm_index.hpp"
#include "utils/include.hpp"
#include "utils/utils.hpp"
#include "utils/LatencyRecorder.hpp"
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
    time_t index_time;
    time_t cache_time;
    size_t height;
    size_t data_IOs;
};


using timer = std::chrono::high_resolution_clock;

// 全局统计
std::atomic<size_t> global_queries_done{0};

// 单线程执行函数
// 单线程执行函数（每个 query 端到端延迟）
template <size_t Epsilon, size_t M>
void worker_thread(falcon::FalconPGM<uint64_t, Epsilon, 4>* F,
                   const std::vector<KeyType>& queries,
                   size_t begin, size_t end,
                   std::atomic<long long>& total_time_ns,
                   LatencyRecorder* latency) {
    using clock_t = std::chrono::high_resolution_clock;
    constexpr size_t BATCH = 128;

    struct Pending {
        size_t qid;
        clock_t::time_point t0;
        std::future<falcon::PointResult> fut;
    };

    auto thread_t0 = clock_t::now();

    std::vector<Pending> inflight;
    inflight.reserve(BATCH);

    size_t i = begin;
    while (i < end) {
        inflight.clear();
        const size_t j = std::min(end, i + BATCH);

        // 提交阶段：为每个 query 记录开始时间 + qid
        for (; i < j; ++i) {
            auto t0 = clock_t::now();                      // 记录每条 query 的“提交时刻”
            auto fut = F->point_lookup(queries[i]);        // 生成 future（不会阻塞）
            inflight.push_back(Pending{ i, t0, std::move(fut) });
        }

        // 回收阶段：逐个 get，并写入该 query 的 latency
        for (auto& p : inflight) {
            (void)p.fut.get();                             // 等待该 query 完成（可在此取 found 等信息）
            // auto t1 = clock_t::now();
            // uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - p.t0).count();
            // latency->set(p.qid, ns);                        // ← 关键：按全局下标写回该 query 的延迟
            global_queries_done.fetch_add(1, std::memory_order_relaxed);
        }
    }

    auto thread_t1 = clock_t::now();
    auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(thread_t1 - thread_t0).count();
    latency->set(begin, dt); 
    total_time_ns.fetch_add(dt, std::memory_order_relaxed);
}


// 多线程 benchmark
template <size_t Epsilon, size_t M>
BenchmarkResult benchmark_mt(std::vector<KeyType> data,
                             std::vector<KeyType> queries,
                             std::string filename,
                             pgm::CachePolicy s,
                             int num_threads) {
    // 1) 只构建 PGM（用于页窗口估计；不让它自己做 I/O/缓存）
    pgm::PGMIndex<KeyType, Epsilon> index(data);

    // 2) 打开数据文件（FALCON 持有 fd）
    int data_fd = ::open(filename.c_str(), O_RDONLY | O_DIRECT);
    if (data_fd < 0) { perror("open data"); std::exit(1); }

    // 3) 策略映射
    pgm::CachePolicy policy = pgm::CachePolicy::NONE;
    switch (s) {
        case pgm::CachePolicy::LRU:  policy = pgm::CachePolicy::LRU;  break;
        case pgm::CachePolicy::FIFO: policy = pgm::CachePolicy::FIFO; break;
        case pgm::CachePolicy::LFU:  policy = pgm::CachePolicy::LFU;  break;
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
        /*cache_shards=*/ 1,                
        /*max_pages_per_batch=*/ 256,       
        /*max_wait_us=*/ 50,                
        /*workers=*/ std::min(std::max(num_threads/8, 1),16)   
    );

    // 5) 多线程提交查询（每线程用批量 futures）
    std::atomic<long long> total_time_ns{0};
    global_queries_done = 0;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    size_t per_thread = queries.size() / num_threads;

    const size_t Q = queries.size();
    LatencyRecorder* latency = new LatencyRecorder(Q);
    auto start_all = timer::now();
    for (int t = 0; t < num_threads; t++) {
        size_t begin = t * per_thread;
        size_t end   = (t == num_threads - 1) ? queries.size() : (t + 1) * per_thread;
        threads.emplace_back(worker_thread<Epsilon, M>, &F, std::ref(queries), begin, end, std::ref(total_time_ns),latency);
    }
    for (auto& th : threads) th.join();
    auto end_all = timer::now();
    auto wall_clock_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_all - start_all).count();

    // latency->dump_csv("./per_query_latency_T" + std::to_string(num_threads) + ".csv",
    //              queries /*, 可选: &thread_of_query, &found */);
    // 6) 从 FALCON 拉统计
    auto st = F.stats();
    double hit_ratio = 0.0;
    auto hm = st.cache_hits + st.cache_misses;
    if (hm) hit_ratio = double(st.cache_hits) / double(hm);

    BenchmarkResult result;
    result.epsilon       = Epsilon;
    result.time_ns       = latency->get_avg();  // 平均 query latency（墙钟）
    result.hit_ratio     = hit_ratio;
    result.total_time    = wall_clock_ns;    // ns
    result.data_IO_time  = st.io_ns;         // FALCON 收集的 I/O 时间累计（ns）
    result.height        = index.height();   // PGM 高度
    result.data_IOs      = st.physical_ios;  // 物理 I/O 片段数（合并后）
    result.index_time    = st.index_ns;
    result.cache_time    = st.cache_ns;
    ::close(data_fd);
    return result;
}

int main() {
    std::string dataset = "osm_cellids";
    std::string filename = "osm_cellids_10M_uint64_unique";
    std::string query_filename = "osm_cellids_10M_uint64_unique.query.bin";
    std::string file = DATASETS + filename;         
    std::string query_file = DATASETS + query_filename;

    std::vector<KeyType> data    = load_data(file, 10000000);
    std::vector<KeyType> queries = load_queries(query_file);

    size_t repeat = 5;
    const size_t MemoryBudget = 40ull * 1024 * 1024;
    // const size_t MemoryBudget = 0;

    std::ofstream csv("osm_falcon_test.csv", std::ios::out | std::ios::trunc);
    if (!csv) {
        std::cerr << "Failed to open CSV output\n";
        return 1;
    }
    csv << "threads,avg_latency_ns,avg_walltime_s,avg_IOs,data_IO_time\n";
    csv << std::fixed << std::setprecision(6);
    for (int r=0; r<repeat; ++r){
        for (int e = 0; e <= 14; ++e) {
            uint64_t threads = 1ULL << e;
            auto result = benchmark_mt<16, MemoryBudget>(data, queries, file, pgm::CachePolicy::NONE, threads);

            std::cout << "[Threads=" << threads << "] ε=" << result.epsilon
                    << ", avg query time=" << result.time_ns << " ns"
                    << ", hit ratio=" << result.hit_ratio
                    << ", total wall time=" << result.total_time / 1e9 << " s"
                    << ", data IOs=" << result.data_IOs 
                    << ", data IO time=" << result.data_IO_time / 1e9 << " s" 
                    << ", index time=" << result.index_time << " ns"
                    << ", cache time=" << result.cache_time << " ns" << std::endl;
            csv << threads << "," << result.time_ns << "," << result.total_time  << "," << result.data_IOs << ","
            << result.data_IO_time << "\n";
            csv.flush();
        }
    }
    csv.close();
    return 0;
}
