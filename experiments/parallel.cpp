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
#include "distribution/zipf.hpp"
#include "pgm/pgm_index_cost.hpp"
#include "utils/include.hpp"
#include "utils/utils.hpp"

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
    double index_hit_ratio;
    time_t total_time;
    time_t data_IO_time;
    time_t index_IO_time;
    size_t height;
    size_t data_IOs;
};


using timer = std::chrono::high_resolution_clock;

// 全局统计
std::atomic<size_t> global_queries_done{0};

// 单线程执行函数
template <size_t Epsilon, size_t M>
void worker_thread(pgm::PGMIndex<KeyType, Epsilon, M, pgm::CacheType::DATA>* index,
                   const std::vector<KeyType>& queries,
                   size_t begin, size_t end,
                   std::atomic<long long>& total_time_ns) {
    auto t0 = timer::now();

    for (size_t i = begin; i < end; i++) {
        auto q = queries[i];
        auto range = index->search(q, pgm::ONE_BY_ONE);
        std::vector<pgm::Record> records = range.records;
        size_t lo = range.lo;
        size_t hi = range.hi;
        binary_search_record(records.data(), lo, hi, q);
        global_queries_done.fetch_add(1, std::memory_order_relaxed);
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
    pgm::PGMIndex<KeyType, Epsilon, M, pgm::CacheType::DATA> index(data, filename, s);

    std::atomic<long long> total_time_ns{0};
    global_queries_done = 0;

    std::vector<std::thread> threads;
    size_t per_thread = queries.size() / num_threads;

    auto start_all = timer::now();
    for (int t = 0; t < num_threads; t++) {
        size_t begin = t * per_thread;
        size_t end = (t == num_threads - 1) ? queries.size() : (t + 1) * per_thread;
        threads.emplace_back(worker_thread<Epsilon, M>, &index, std::ref(queries), begin, end, std::ref(total_time_ns));
    }

    for (auto& th : threads) th.join();
    auto end_all = timer::now();

    auto wall_clock_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_all - start_all).count();

    auto cache = index.get_data_cache();
    auto index_cache = index.get_index_cache();

    BenchmarkResult result;
    result.epsilon = Epsilon;
    result.time_ns = (double)wall_clock_ns / queries.size();  // 平均 query latency
    result.hit_ratio = (double)cache->get_hit_count() / (cache->get_hit_count() + cache->get_miss_count());
    result.index_hit_ratio = (double)index_cache->get_hit_count() / (index_cache->get_hit_count() + index_cache->get_miss_count());
    result.total_time = wall_clock_ns;     // 墙钟时间
    result.data_IO_time = cache->get_IO_time();
    result.index_IO_time = index_cache->get_IO_time();
    result.height = index.height();
    result.data_IOs = cache->get_IOs();
    return result;
}

int main() {
    std::string dataset = "books";
    std::string filename = "books_20M_uint64_unique";
    std::string query_filename = "books_20M_uint64_unique.10Mquery.bin";
    std::string file = DATASETS + filename;
    std::string query_file = DATASETS + query_filename;

    std::vector<KeyType> data = load_data(file,20000000);
    std::vector<KeyType> queries = load_queries(query_file);

    const size_t MemoryBudget = 40*1024*1024;
    int num_threads = 32;  // 你想要的线程数
    for (int i = 2; i < num_threads; i++) {
        BenchmarkResult result = benchmark_mt<8, MemoryBudget>(data, queries, file, pgm::CacheStrategy::LRU, i);

        std::cout << "[Threads=" << i << "] ε=" << result.epsilon
                << ", avg query time=" << result.time_ns << " ns"
                << ", hit ratio=" << result.hit_ratio
                << ", total wall time=" << result.total_time / 1e9 << " s"
                << ", data IOs=" << result.data_IOs << std::endl;
    }
    

    return 0;
}
