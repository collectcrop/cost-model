/*
 * This example shows how to index and query a vector of random integers with the PGM-index.
 * Compile with:
 *   g++ simple.cpp -std=c++17 -I../include -o simple
 * Run with:
 *   ./simple
 */
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
#include "distribution/zipf.hpp"
#include "pgm/pgm_index.hpp"
#include "utils/utils.hpp"
#include "FALCON/Falcon.hpp"

using KeyType = uint64_t;
#define DIRECTORY "/mnt/home/zwshi/learned-index/cost-model/experiments/"
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"

// struct Record {
//     uint64_t key;
// };

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



template <size_t Epsilon, size_t M>
BenchmarkResult benchmark(std::vector<KeyType> data,
                          std::vector<KeyType> queries,
                          std::string filename,
                          pgm::CacheStrategy s) {
    // 1) 建索引（只做“定位窗口”，不触发 I/O）
    pgm::PGMIndex<KeyType, Epsilon> index(data);

    // 2) 打开数据文件（O_DIRECT）
    int data_fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
    if (data_fd < 0) { perror("open"); std::exit(1); }

    // 3) Cache 策略映射到 FALCON 的策略
    pgm::CachePolicy policy = pgm::CachePolicy::LRU;
    switch (s) {
        case pgm::CacheStrategy::LRU: policy = pgm::CachePolicy::LRU; break;
        case pgm::CacheStrategy::FIFO: policy = pgm::CachePolicy::FIFO; break;
        case pgm::CacheStrategy::LFU: policy = pgm::CachePolicy::LFU; break;
    }

    // 4) FALCON：一个 worker，一个 ring；用你的预算 M 初始化缓存
    falcon::FalconPGM<uint64_t, Epsilon, 4> F(
        index,
        data_fd,
        pgm::IO_URING,
        /*memory_budget_bytes=*/ M,
        /*cache_policy=*/ policy,
        /*cache_shards=*/ 0,          // 2^0
        /*max_pages_per_batch=*/ 128,
        /*max_wait_us=*/ 50,
        /*workers=*/ 1
    );

    // 5) 提交全部查询（让 FALCON 在 worker 侧聚合/去重/批量 I/O）
    std::vector<std::future<falcon::PointResult>> futs;
    futs.reserve(queries.size());

    auto t0 = timer::now();
    for (auto &q : queries) {
        futs.emplace_back(F.point_lookup(q));
    }
    size_t found = 0;
    for (auto &f : futs) {
        auto r = f.get();
        if (r.found) ++found; // 仅示意：你也可以校验正确性
    }
    auto t1 = timer::now();

    // 6) 统计
    auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    double query_ns = double(t) / queries.size();

    auto st = F.stats(); // 从 FALCON 聚合得到命中与 I/O 指标

    BenchmarkResult result;
    result.epsilon    = Epsilon;
    result.time_ns    = query_ns;
    result.hit_ratio  = (st.cache_hits + st.cache_misses)
                        ? double(st.cache_hits) / double(st.cache_hits + st.cache_misses)
                        : 0.0;
    result.total_time = t;                // ns
    result.data_IO_time = st.io_ns;       // ns（批量 I/O 时间累计）
    result.height     = index.height();
    result.data_IOs   = st.physical_ios;  // 物理读取次数（聚合后）
    close(data_fd);
    return result;
}


std::string mk_outfilename(pgm::CacheStrategy s,std::string dataset,size_t ds, size_t ms, std::string suffix=""){
    std::string prefix;
    switch (s){
        case pgm::CacheStrategy::LRU:
            prefix = "LRU-";
            break;
        case pgm::CacheStrategy::FIFO:
            prefix = "FIFO-";
            break;
        case pgm::CacheStrategy::LFU:
            prefix = "LFU-";
            break;
    }
    return prefix + dataset + "-" + std::to_string(ds) + "M-M" + std::to_string(ms) + suffix +".csv";
}

int main() {
    // cpu_set_t mask;
    // CPU_ZERO(&mask);
    // CPU_SET(13, &mask);   // bind to CPU 13

    // if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
    //     perror("sched_setaffinity");
    //     return 1;
    // }

    std::string dataset = "books";
    std::string filename = "books_10M_uint64_unique";
    std::string query_filename = "books_10M_uint64_unique.query.bin";
    std::string file = DATASETS + filename;
    std::string query_file = DATASETS + query_filename;
    std::vector<KeyType> data = load_data(file,10000000);
    std::vector<KeyType> queries = load_queries(query_file);
    const size_t MemoryBudget = 20*1024*1024;
    
    int trials = 10;
    for (pgm::CacheStrategy s: {pgm::CacheStrategy::LRU}){     //pgm::CacheStrategy::LRU,pgm::CacheStrategy::FIFO,pgm::CacheStrategy::LFU
        std::ofstream ofs(mk_outfilename(s,dataset,20,MemoryBudget>>20,".5"));
        ofs << "epsilon,avg_query_time_ns,avg_cache_hit_ratio,data_IOs,total_time\n";
        for (int i=0;i<trials;i++){
            for (size_t epsilon : {8}) {     //8,10,12,14,16,18,20,24,32,48,64,128,256
                BenchmarkResult result;
                switch (epsilon) {
                    case 2: result = benchmark<2, MemoryBudget>(data, queries, file, s); break;
                    case 4: result = benchmark<4, MemoryBudget>(data, queries, file, s); break;
                    case 6: result = benchmark<6, MemoryBudget>(data, queries, file, s); break;
                    case 8:  result = benchmark<8, MemoryBudget>(data, queries, file, s); break;
                    case 9: result = benchmark<9, MemoryBudget>(data, queries, file, s); break;
                    case 10: result = benchmark<10, MemoryBudget>(data, queries, file, s); break;
                    case 11: result = benchmark<11, MemoryBudget>(data, queries, file, s); break;
                    case 12: result = benchmark<12, MemoryBudget>(data, queries, file, s); break;
                    case 13: result = benchmark<13, MemoryBudget>(data, queries, file, s); break;
                    case 14: result = benchmark<14, MemoryBudget>(data, queries, file, s); break;
                    case 15: result = benchmark<15, MemoryBudget>(data, queries, file, s); break;
                    case 16: result = benchmark<16, MemoryBudget>(data, queries, file, s); break;
                    case 17: result = benchmark<17, MemoryBudget>(data, queries, file, s); break;
                    case 18: result = benchmark<18, MemoryBudget>(data, queries, file, s); break;
                    case 19: result = benchmark<19, MemoryBudget>(data, queries, file, s); break;
                    case 20: result = benchmark<20, MemoryBudget>(data, queries, file, s); break;
                    case 21: result = benchmark<21, MemoryBudget>(data, queries, file, s); break;
                    case 22: result = benchmark<22, MemoryBudget>(data, queries, file, s); break;
                    case 23: result = benchmark<23, MemoryBudget>(data, queries, file, s); break;
                    case 24: result = benchmark<24, MemoryBudget>(data, queries, file, s); break;
                    case 25: result = benchmark<25, MemoryBudget>(data, queries, file, s); break;
                    case 26: result = benchmark<26, MemoryBudget>(data, queries, file, s); break;
                    case 28: result = benchmark<28, MemoryBudget>(data, queries, file, s); break;
                    case 30: result = benchmark<30, MemoryBudget>(data, queries, file, s); break;
                    case 32: result = benchmark<32, MemoryBudget>(data, queries, file, s); break;
                    case 40: result = benchmark<40, MemoryBudget>(data, queries, file, s); break;
                    case 48: result = benchmark<48, MemoryBudget>(data, queries, file, s); break;
                    case 56: result = benchmark<56, MemoryBudget>(data, queries, file, s); break;
                    case 64: result = benchmark<64, MemoryBudget>(data, queries, file, s); break;
                    case 128: result = benchmark<128, MemoryBudget>(data, queries, file, s); break;
                    case 256: result = benchmark<256, MemoryBudget>(data, queries, file, s); break;
                }

                std::cout << "ε=" << result.epsilon
                        << ", time=" << result.time_ns << " ns"
                        << ", hit ratio=" << result.hit_ratio
                        << ", total time=" << result.total_time / 1e9 << " s"
                        << ", data IO time=" << result.data_IO_time << " ns"
                        << ", data IOs=" << result.data_IOs
                        << ", height=" << result.height << std::endl;
                ofs << result.epsilon << "," << result.time_ns << "," << result.hit_ratio << "," 
                << result.data_IOs <<  "," << result.total_time <<"\n";
            }
        }
        
        ofs.close();
    }
    
    return 0;
}


