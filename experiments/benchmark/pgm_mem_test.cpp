// pgm_mem_test.cpp
#include <cstdlib>
#include <sstream> 
#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h> 
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>

#include "distribution/zipf.hpp"
#include "pgm/pgm_index.hpp"
#include "utils/utils.hpp"   // 里面有 load_data_pgm_safe / load_queries_pgm_safe

using KeyType = uint64_t;
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"
#define Epsilon 76
#define EpsilonRecur 32

struct BenchmarkResult {
    size_t epsilon;
    double time_ns;      // 平均 query latency (ns)
    double total_time_s; // wall time (秒)
    size_t data_IOs;     // 内存模型固定为 0
    size_t height;
    uint64_t index_cpu_ns;  
    uint64_t io_wait_ns; 
};

using timer = std::chrono::high_resolution_clock;

struct ThreadArgs {
    std::vector<KeyType>* data;
    std::vector<KeyType>* queries;
    pgm::PGMIndex<KeyType, Epsilon, EpsilonRecur>* index;
    size_t start;
    size_t end;
    BenchmarkResult result;
};

// 单线程查询：PGM + 内存二分
void* query_worker(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;

    auto* data    = args->data;
    auto* queries = args->queries;
    auto* index   = args->index;

    auto t0 = timer::now();

    for (size_t i = args->start; i < args->end; i++) {
        KeyType q = (*queries)[i];

        auto range = index->search(q);
        size_t lo = range.lo;
        size_t hi = range.hi;

        // 边界保护
        if (lo >= data->size()) lo = data->size() - 1;
        if (hi >= data->size()) hi = data->size() - 1;
        if (lo > hi) std::swap(lo, hi);

        // 在 [lo, hi] 上做二分查找
        auto it = std::lower_bound(data->begin() + lo,
                                   data->begin() + hi + 1,
                                   q);
        bool found = (it != data->end() && *it == q);
        if (!found) std::cout << "Not found: " << q << "\n";
    }

    auto t1 = timer::now();
    auto t  = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    args->result.time_ns      = (double)t / (args->end - args->start);
    args->result.total_time_s = t / 1e9;
    args->result.data_IOs     = 0;                // 内存模型，不产生磁盘 IO
    args->result.height       = index->height();
    args->result.index_cpu_ns = t;                // 这里就等于线程总 CPU 时间
    args->result.io_wait_ns   = 0;                // 没有 I/O 等待

    return nullptr;
}

// 多线程 benchmark：和 pgm-disk-test 的 run_experiment 形式类似
BenchmarkResult run_experiment(std::vector<KeyType>& data,
                               std::vector<KeyType>& queries,
                               pgm::PGMIndex<KeyType, Epsilon, EpsilonRecur>& index,
                               int THREADS) {
    size_t total_queries = queries.size();
    size_t per_thread    = total_queries / THREADS;

    pthread_t   threads[THREADS];
    ThreadArgs  args[THREADS];

    auto t0 = timer::now();

    for (int i = 0; i < THREADS; i++) {
        args[i].data    = &data;
        args[i].queries = &queries;
        args[i].index   = &index;
        args[i].start   = i * per_thread;
        args[i].end     = (i == THREADS - 1) ? total_queries : (i + 1) * per_thread;

        pthread_create(&threads[i], nullptr, query_worker, &args[i]);
    }

    for (int i = 0; i < THREADS; i++) {
        pthread_join(threads[i], nullptr);
    }

    auto t1        = timer::now();
    auto total_ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // 汇总结果（average over threads）
    BenchmarkResult r{};
    r.epsilon       = Epsilon;
    r.total_time_s  = total_ns / 1e9;
    r.height        = index.height();
    r.data_IOs      = 0;
    r.time_ns       = 0.0;
    r.index_cpu_ns  = 0;
    r.io_wait_ns    = 0;

    for (int i = 0; i < THREADS; i++) {
        r.time_ns      += args[i].result.time_ns;
        r.index_cpu_ns += args[i].result.index_cpu_ns;
        // data_IOs、io_wait_ns 都是 0，不用加
    }
    r.time_ns /= THREADS;

    return r;
}

int main(int argc, char** argv) {
    // helper
    if (argc < 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " <dataset_basename> <num_keys> [max_log2_threads] [baseline_name]\n\n"
                  << "Example:\n  " << argv[0]
                  << " books_200M_uint64_unique 200000000 10 PGM-mem\n";
        return 1;
    }

    // 1) essential parameter
    std::string dataset_basename = argv[1];              // 如 books_200M_uint64_unique
    uint64_t num_keys            = std::strtoull(argv[2], nullptr, 10);

    // 2) max log2(threads)
    int max_exp = 10;  // 默认 2^0 .. 2^10
    if (argc >= 4) {
        max_exp = std::atoi(argv[3]);
        if (max_exp < 0) max_exp = 0;
    }

    // 3) baseline name
    std::string baseline = "PGM-mem";
    if (argc >= 5) {
        baseline = argv[4];
    }

    // 4) 拼接文件路径
    std::string filename       = dataset_basename;                // books_200M_uint64_unique
    std::string query_filename = dataset_basename + ".query.bin"; // books_200M_uint64_unique.query.bin
    std::string file       = DATASETS + filename;
    std::string query_file = DATASETS + query_filename;

    // 5) 加载数据（整块载入内存）
    std::vector<KeyType> data    = load_data_pgm_safe<KeyType>(file, num_keys);
    std::vector<KeyType> queries = load_queries_pgm_safe<KeyType>(query_file);

    if (data.size() == 0 || queries.size() == 0) {
        std::cerr << "load_data or load_queries failed, data=" << data.size()
                  << ", queries=" << queries.size() << "\n";
        return 1;
    }

    // 6) 构建 PGM 索引（完全内存）
    pgm::PGMIndex<KeyType, Epsilon, EpsilonRecur> index(data);

    // 7) 输出文件名：<basename>_pgm_mem_multithread.csv
    std::string out_csv = dataset_basename + "_pgm_mem_multithread.csv";
    std::ofstream ofs(out_csv);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output csv: " << out_csv << "\n";
        return 1;
    }

    ofs << "baseline,threads,latency_ns,walltime_s,height,avg_IOs\n";

    // 8) 线程实验循环，与磁盘版保持一致
    for (int exp = 0; exp <= max_exp; exp++) {
        int threads = 1 << exp;

        double avg_latency = 0;
        double avg_wall    = 0;
        size_t avg_IOs     = 0;
        size_t height      = 0;

        // 简单做 3 次复跑取平均
        const int RUNS = 3;
        for (int t = 0; t < RUNS; t++) {
            BenchmarkResult r = run_experiment(data, queries, index, threads);
            avg_latency += r.time_ns;
            avg_wall    += r.total_time_s;
            avg_IOs     += r.data_IOs;
            height       = r.height;
        }
        avg_latency /= RUNS;
        avg_wall    /= RUNS;
        avg_IOs     /= RUNS;

        std::cout << "threads=" << threads
                  << ", avg_latency=" << avg_latency << " ns"
                  << ", wall=" << avg_wall << " s"
                  << ", IOs=" << avg_IOs
                  << ", height=" << height << std::endl;

        ofs << baseline << "," << threads << "," << avg_latency << ","
            << avg_wall   << "," << height  << "," << avg_IOs << "\n";
    }

    ofs.close();
    return 0;
}
