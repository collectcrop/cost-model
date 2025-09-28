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
#include "pgm/pgm_index_cost.hpp"
#include "utils/utils.hpp"

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
BenchmarkResult benchmark(std::vector<KeyType> data,std::vector<KeyType> queries,std::string filename, pgm::CacheStrategy s) {
    pgm::PGMIndexCost<KeyType, Epsilon, M, pgm::CacheType::DATA> index(data,filename,s);
    auto t0 = timer::now();
    int cnt = 0;
    for (auto &q : queries) {
        auto range = index.search(q, pgm::ALL_IN_ONCE);
        std::vector<pgm::Record> records = range.records;
        size_t lo = range.lo;
        size_t hi = range.hi;
        const bool result = binary_search_record(records.data(), lo, hi, q);
        // if (result) cnt++;
    }
    // std::cout << "total: " << cnt << std::endl;
    auto t1 = timer::now();
    auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    auto query_ns = t / queries.size();
    
    auto cache = index.get_data_cache();

    size_t total_hits = cache->get_hit_count();
    size_t total_access = cache->get_hit_count() + cache->get_miss_count();
    std::cout << "C=" << cache->get_C() << std::endl; 
    BenchmarkResult result;
    result.epsilon = Epsilon;
    result.time_ns = query_ns;
    result.hit_ratio = static_cast<double>(total_hits) / total_access;
    result.total_time = t;
    result.data_IO_time = cache->get_IO_time();
    result.height = index.height();
    result.data_IOs = cache->get_IOs();
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
    std::string filename = "books_20M_uint64_unique";
    std::string query_filename = "books_20M_uint64_unique.100Ktable2.bin";
    std::string file = DATASETS + filename;
    std::string query_file = DATASETS + query_filename;
    std::vector<KeyType> data = load_data(file,20000000);
    std::vector<KeyType> queries = load_queries(query_file);
    const size_t MemoryBudget = 20*1024*1024;
    
    int trials = 10;
    for (pgm::CacheStrategy s: {pgm::CacheStrategy::LRU,pgm::CacheStrategy::FIFO,pgm::CacheStrategy::LFU}){     //pgm::CacheStrategy::LRU,pgm::CacheStrategy::FIFO,pgm::CacheStrategy::LFU
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


