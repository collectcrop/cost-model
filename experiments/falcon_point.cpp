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
#include "FALCON/Falcon.hpp"     
#include "cache/CacheInterface.hpp" 

using KeyType = uint64_t;
#define DIRECTORY "/mnt/home/zwshi/learned-index/cost-model/experiments/"

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
// 单线程执行函数（每个 query 端到端延迟）
template <size_t Epsilon>
void worker_thread(falcon::FalconPGM<uint64_t, Epsilon, 4>* F,
                   const std::vector<KeyType>& queries,
                   size_t begin, size_t end,
                   std::atomic<long long>& total_time_ns,
                   LatencyRecorder* latency,
                   size_t batch_size) {
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
            auto fut = p.fut.get();                             // 等待该 query 完成（可在此取 found 等信息）
            if (!fut.found) std::cout << "not found" << std::endl;
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
template <size_t Epsilon>
BenchmarkResult benchmark_mt(std::vector<KeyType> data,
                             std::vector<KeyType> queries,
                             std::string filename,
                             falcon::CachePolicy s,
                             int num_threads,
                             size_t M,
                             size_t batch_size = 128) {
    // 1) 只构建 PGM（用于页窗口估计；不让它自己做 I/O/缓存）
    pgm::PGMIndex<KeyType, Epsilon> index(data);

    // 2) 打开数据文件（FALCON 持有 fd）
    int data_fd = ::open(filename.c_str(), O_RDONLY | O_DIRECT);
    if (data_fd < 0) { perror("open data"); std::exit(1); }

    // 3) 策略映射
    falcon::CachePolicy policy = falcon::CachePolicy::NONE;
    switch (s) {
        case falcon::CachePolicy::LRU:  policy = falcon::CachePolicy::LRU;  break;
        case falcon::CachePolicy::FIFO: policy = falcon::CachePolicy::FIFO; break;
        case falcon::CachePolicy::LFU:  policy = falcon::CachePolicy::LFU;  break;
    }

    // 4) 构建 FALCON 引擎
    //    - MemoryBudget 用运行时参数 M（字节）
    //    - workers = num_threads（也可先设 1，简化确定性；两种都可）
    falcon::FalconPGM<uint64_t, Epsilon, /*EpsRec*/ 4> F(
        index,
        data_fd,
        falcon::IO_URING,
        /*memory_budget_bytes=*/ M,
        /*cache_policy=*/ policy,
        /*cache_shards=*/ 1,                
        /*max_pages_per_batch=*/ 256,       
        /*max_wait_us=*/ 50,                
        /*workers=*/ std::max(num_threads/16, 1)   
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
    std::sort(queries.begin(), queries.end());
    // auto test = timer::now();
    // std::cout << "sort time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(test - start_all).count() << std::endl;
    for (int t = 0; t < num_threads; t++) {
        size_t begin = t * per_thread;
        size_t end   = (t == num_threads - 1) ? queries.size() : (t + 1) * per_thread;
        threads.emplace_back(worker_thread<Epsilon>, &F, std::ref(queries), begin, end, std::ref(total_time_ns),latency, batch_size);
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
    result.data_IOs      = st.physical_ios;  // logical_ios  
    ::close(data_fd);
    return result;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " <dataset_basename> <num_keys> [memory_mb] [repeats]\n\n"
                  << "Example:\n  " << argv[0]
                  << " books_200M_uint64_unique 200000000 40 3\n";
        return 1;
    }

    // 1) 必选参数
    std::string dataset_basename = argv[1]; 
    uint64_t num_keys = std::strtoull(argv[2], nullptr, 10);

    // 2) 可选: memory budget (MB)
    int mem_mb = 256;
    if (argc >= 4) {
        mem_mb = std::atoi(argv[3]);
        if (mem_mb < 0) mem_mb = 0;
    }
    size_t MemoryBudget = static_cast<size_t>(mem_mb) * 1024ull * 1024ull;
    
    // 3) 可选: repeats
    size_t repeats = 10;
    if (argc >= 5) {
        repeats = std::atoi(argv[4]);
        if (repeats <= 0) repeats = 1;
    }

    std::string filename     = dataset_basename;                  // e.g. books_200M_uint64_unique
    std::string query_fname  = dataset_basename + ".1Mtable1.bin";  
    std::string file         = falcon::DATASETS + filename;
    std::string query_file   = falcon::DATASETS + query_fname;

    std::vector<KeyType> data    = load_data_pgm_safe<KeyType>(file, num_keys);
    std::vector<KeyType> queries = load_queries_pgm_safe<KeyType>(query_file);

    std::ofstream csv("books-200M-point.csv", std::ios::out | std::ios::trunc);
    if (!csv) {
        std::cerr << "Failed to open CSV output\n";
        return 1;
    }
    csv << "epsilon,"<< "avg_latency_ns,"<< "total_wall_time_s,"<< "avg_IOs,"<< "IO_time_s,"<< "mem_time_s,"
    << "IO_fraction,"<< "mem_fraction,"<< "cache_hit_ratio\n";
    csv << std::fixed << std::setprecision(6);
    uint64_t threads = 1;
    falcon::CachePolicy s = falcon::CachePolicy::LRU;
    for (int i=0;i<repeats;i++){
        size_t epsilon = 16;
        BenchmarkResult result;
        // if (MemoryBudget<16*num_keys/(2*epsilon)){
        //     std::cout << "Memory budget too small for ε=" << epsilon << ", skipping.\n";
        //     continue;
        // }
        // const size_t M = MemoryBudget - 16*num_keys/(2*epsilon);
        const size_t M = MemoryBudget;
        result = benchmark_mt<16>(data, queries, file, s, threads, M);
        
        double total_time_s   = result.total_time / 1e9;
        double io_time_s      = result.data_IO_time / 1e9;
        double mem_time_s     = (result.total_time - result.data_IO_time) / 1e9;
        double io_ratio       = double(result.data_IO_time) / double(result.total_time);
        double mem_ratio      = 1.0 - io_ratio;
        std::cout << "[Threads=" << threads << "] ε=" << result.epsilon
            << ", avg query time=" << result.time_ns << " ns"
            << ", hit ratio=" << result.hit_ratio
            << ", total wall time=" << total_time_s << " s"
            << ", IO time=" << io_time_s << " s"
            << ", Mem(other) time=" << mem_time_s << " s"
            << ", IO fraction=" << io_ratio
            << ", Mem fraction=" << mem_ratio
            << ", data IOs=" << result.data_IOs
            << std::endl;
        csv << epsilon << ","
            << result.time_ns << ","
            << total_time_s << ","
            << result.data_IOs << ","
            << io_time_s << ","
            << mem_time_s << ","
            << io_ratio << ","
            << mem_ratio << ","
            << result.hit_ratio
            << "\n";
        csv.flush();
        
    }
    
    csv.close();
    return 0;
}
