// test producer and worker thread configuration
#include <sched.h>
#include <cstdlib>
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
    size_t height;
    size_t data_IOs;
};


using timer = std::chrono::high_resolution_clock;

// global variables for tracking progress and configuration
std::atomic<size_t> global_queries_done{0};
std::atomic<int> g_falcon_workers{1};

template <size_t Epsilon>
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

        // submission phase
        for (; i < j; ++i) {
            auto t0 = clock_t::now();                      // record submission time
            auto fut = F->point_lookup(queries[i]);        // trigger the query and get a future
            inflight.push_back(Pending{ i, t0, std::move(fut) });
        }

        // completion phase
        for (auto& p : inflight) {
            (void)p.fut.get();                             // wait for completion (result is ignored in this benchmark)
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


// benchmark
template <size_t Epsilon>
BenchmarkResult benchmark_mt(std::vector<KeyType> data,
                             std::vector<KeyType> queries,
                             const std::string &filename,
                             falcon::CachePolicy s,
                             int num_threads,
                             size_t memory_budget_bytes) {
    // 1) construct PGM
    pgm::PGMIndex<KeyType, Epsilon> index(data);

    // 2) open data file
    int data_fd = ::open(filename.c_str(), O_RDONLY|O_DIRECT);
    if (data_fd < 0) {
        perror("open data");
        std::exit(1);
    }

    // 3) configure cache policy
    falcon::CachePolicy policy = falcon::CachePolicy::NONE;
    switch (s) {
        case falcon::CachePolicy::LRU:  policy = falcon::CachePolicy::LRU;  break;
        case falcon::CachePolicy::FIFO: policy = falcon::CachePolicy::FIFO; break;
        case falcon::CachePolicy::LFU:  policy = falcon::CachePolicy::LFU;  break;
        case falcon::CachePolicy::NONE: policy = falcon::CachePolicy::NONE; break;
    }

    int workers = g_falcon_workers.load(std::memory_order_relaxed);

    // 4) initialize FALCON
    falcon::FalconPGM<uint64_t, Epsilon, /*EpsRec*/ 4> F(
        index,
        data_fd,
        falcon::IO_URING,
        /*memory_budget_bytes=*/ memory_budget_bytes,
        /*cache_policy=*/ policy,
        /*cache_shards=*/ 4,
        /*max_pages_per_batch=*/ 256,
        /*max_wait_us=*/ 50,
        /*workers=*/ workers          
    );

    // 5) spawn worker threads to issue queries
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
                             begin, end, std::ref(total_time_ns), latency);
    }
    for (auto &th : threads) th.join();
    auto end_all = timer::now();
    auto wall_clock_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_all - start_all).count();

    // 6) gather stats and compute results
    auto st = F.stats();
    double hit_ratio = 0.0;
    auto hm = st.cache_hits + st.cache_misses;
    if (hm) hit_ratio = double(st.cache_hits) / double(hm);

    BenchmarkResult result;
    result.epsilon      = Epsilon;
    result.time_ns      = latency->get_avg();
    result.hit_ratio    = hit_ratio;
    result.total_time   = wall_clock_ns;
    // result.data_IO_time = st.io_ns;
    result.height       = index.height();
    result.data_IOs     = st.physical_ios;
    // result.index_time   = st.index_ns;
    // result.cache_time   = st.cache_ns;

    ::close(data_fd);
    delete latency;
    return result;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " <dataset_basename> <num_keys> [repeats] [memory_mb] [baseline_name]\n\n"
                  << "Example:\n  " << argv[0]
                  << " books_200M_uint64_unique 200000000 3 40 FALCON\n";
        return 1;
    }

    std::string dataset_basename = argv[1]; 
    uint64_t num_keys = std::strtoull(argv[2], nullptr, 10);

    int repeats = 3;
    if (argc >= 4) {
        repeats = std::atoi(argv[3]);
        if (repeats <= 0) repeats = 1;
    }

    double mem_mb = 256;
    if (argc >= 5) {
        mem_mb = std::atof(argv[4]);
        if (mem_mb < 0) mem_mb = 0;
    }
    size_t memory_budget_bytes = static_cast<size_t>(mem_mb * 1024ull * 1024ull);

    std::string baseline = "FALCON";
    if (argc >= 6) {
        baseline = argv[5];
    }

    std::cout << "Memory budget: " << mem_mb << " MB (" << memory_budget_bytes << " bytes)\n";
    std::string filename     = dataset_basename;                  // e.g. books_200M_uint64_unique
    std::string query_fname  = dataset_basename + ".query.bin";   // e.g. books_200M_uint64_unique.query.bin
    std::string file         = falcon::DATASETS + filename;
    std::string query_file   = falcon::DATASETS + query_fname;

    std::vector<KeyType> data    = load_data_pgm_safe<KeyType>(file, num_keys);
    std::vector<KeyType> queries = load_queries_pgm_safe<KeyType>(query_file);

    std::string csv_name = dataset_basename + "_falcon_test.csv";
    std::ofstream csv(csv_name, std::ios::out | std::ios::trunc);
    if (!csv) {
        std::cerr << "Failed to open CSV output: " << csv_name << "\n";
        return 1;
    }
    csv << "baseline,producers,workers,latency_ns,wall_time_ns,avg_IOs,hit_ratio\n";
    csv << std::fixed << std::setprecision(6);

    std::vector<int> P_list = {1, 2, 4, 8, 16, 32, 64, 128};
    std::vector<int> W_list = {1, 2, 4, 8, 16, 32, 64, 128};

    for (int r = 0; r < repeats; ++r) {
        for (int W : W_list) {
            g_falcon_workers.store(W, std::memory_order_relaxed);

            for (int P : P_list) {
                int producers = P;

                auto result = benchmark_mt<16>(data, queries, file,
                                            falcon::CachePolicy::LRU,
                                            producers,
                                            memory_budget_bytes);

                std::cout << "[P=" << producers << ", W=" << W << "] ε=" << result.epsilon
                        << ", avg query time=" << result.time_ns << " ns"
                        << ", hit ratio=" << result.hit_ratio
                        << ", total wall time=" << result.total_time / 1e9 << " s"
                        << ", data IOs=" << result.data_IOs
                        << std::endl;

                csv << baseline << ","
                    << producers << ","
                    << W << ","
                    << result.time_ns << ","
                    << result.total_time << ","
                    << result.data_IOs << ","
                    << result.hit_ratio
                    << "\n";
                csv.flush();
            }
        }
    }


    csv.close();
    std::cout << "Finished. Results saved to " << csv_name << "\n";
    return 0;
}