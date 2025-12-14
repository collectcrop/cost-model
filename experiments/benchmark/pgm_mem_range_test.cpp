// pgm_mem_range_test.cpp
// Compile:
//   g++ -O3 -std=c++17 pgm_mem_range_test.cpp -o pgm_mem_range_test -pthread
//
// Usage:
//   ./pgm_mem_range_test <dataset_basename> <num_keys> [max_log2_threads] [baseline_name]
//
// Example:
//   ./pgm_mem_range_test books_200M_uint64_unique 200000000 10 PGM-mem-range

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
#include <algorithm>

#include "pgm/pgm_index.hpp"
#include "utils/utils.hpp"   // load_data_pgm_safe / load_ranges_pgm_safe / pgm::RangeQ

using KeyType = uint64_t;
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"

// 和原来的 pgm_mem_test 保持一致
#define Epsilon      76
#define EpsilonRecur 32

struct BenchmarkResult {
    size_t   epsilon;
    double   time_ns;        // 平均 query latency (ns)
    double   total_time_s;   // wall time (s)
    size_t   data_IOs;       // 内存模型固定为 0
    size_t   height;
    uint64_t index_cpu_ns;   // 这里就等于线程总 CPU 时间
    uint64_t io_wait_ns;     // 0
};

using timer = std::chrono::high_resolution_clock;

struct ThreadArgs {
    std::vector<KeyType>* data;
    std::vector<pgm::RangeQ>* ranges;
    pgm::PGMIndex<KeyType, Epsilon, EpsilonRecur>* index;
    size_t start;
    size_t end;
    BenchmarkResult result;
};

// 单线程 range 查询：PGM + 内存区间二分
void* range_worker(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;

    auto* data   = args->data;
    auto* ranges = args->ranges;
    auto* index  = args->index;

    const size_t n = data->size();

    auto t0 = timer::now();

    for (size_t i = args->start; i < args->end; i++) {
        const auto& rq = (*ranges)[i];
        KeyType lo_key = rq.lo;
        KeyType hi_key = rq.hi;
        if (lo_key > hi_key) std::swap(lo_key, hi_key);

        // 1) 通过 PGM 估计 lo, hi 位置区间
        auto range_lo = index->search(lo_key);
        auto range_hi = index->search(hi_key);

        size_t search_lo = range_lo.lo;
        size_t search_hi = range_hi.hi;

        if (search_lo >= n) search_lo = n - 1;
        if (search_hi >= n) search_hi = n - 1;
        if (search_lo > search_hi) std::swap(search_lo, search_hi);

        // 2) 在 [search_lo, search_hi] 上用 lower_bound / upper_bound 精确定位
        auto it_lo = std::lower_bound(
            data->begin() + search_lo,
            data->begin() + search_hi + 1,
            lo_key
        );
        auto it_hi = std::upper_bound(
            it_lo,
            data->begin() + search_hi + 1,
            hi_key
        );

        // 结果个数（这里只是防止编译器优化掉，真实系统可以按需处理结果）
        size_t cnt = (it_hi > it_lo) ? (size_t)(it_hi - it_lo) : 0;
        (void)cnt;
    }

    auto t1 = timer::now();
    auto t  = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    size_t q_num = (args->end > args->start) ? (args->end - args->start) : 1;

    args->result.time_ns      = (double)t / (double)q_num;
    args->result.total_time_s = t / 1e9;
    args->result.data_IOs     = 0;              // 内存模型
    args->result.height       = index->height();
    args->result.index_cpu_ns = t;
    args->result.io_wait_ns   = 0;

    return nullptr;
}

// 多线程 benchmark：和 pgm_mem_test 的 run_experiment 类似，但用 RangeQ
BenchmarkResult run_experiment_range(std::vector<KeyType>& data,
                                     std::vector<pgm::RangeQ>& ranges,
                                     pgm::PGMIndex<KeyType, Epsilon, EpsilonRecur>& index,
                                     int THREADS) {
    size_t total_queries = ranges.size();
    if (THREADS <= 0) THREADS = 1;
    if ((size_t)THREADS > total_queries) THREADS = (int)total_queries;

    size_t per_thread = total_queries / THREADS;

    pthread_t  threads[THREADS];
    ThreadArgs args[THREADS];

    auto t0 = timer::now();

    for (int i = 0; i < THREADS; i++) {
        args[i].data   = &data;
        args[i].ranges = &ranges;
        args[i].index  = &index;
        args[i].start  = i * per_thread;
        args[i].end    = (i == THREADS - 1) ? total_queries : (i + 1) * per_thread;

        pthread_create(&threads[i], nullptr, range_worker, &args[i]);
    }

    for (int i = 0; i < THREADS; i++) {
        pthread_join(threads[i], nullptr);
    }

    auto t1       = timer::now();
    auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    BenchmarkResult r{};
    r.epsilon      = Epsilon;
    r.total_time_s = total_ns / 1e9;
    r.height       = index.height();
    r.data_IOs     = 0;
    r.time_ns      = 0.0;
    r.index_cpu_ns = 0;
    r.io_wait_ns   = 0;

    for (int i = 0; i < THREADS; i++) {
        r.time_ns      += args[i].result.time_ns;
        r.index_cpu_ns += args[i].result.index_cpu_ns;
    }
    r.time_ns /= THREADS;

    return r;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " <dataset_basename> <num_keys> [max_log2_threads] [baseline_name]\n\n"
                  << "Example:\n  " << argv[0]
                  << " books_200M_uint64_unique 200000000 10 PGM-mem-range\n";
        return 1;
    }

    // 1) 参数
    std::string dataset_basename = argv[1];              
    uint64_t num_keys            = std::strtoull(argv[2], nullptr, 10);

    int max_exp = 10;
    if (argc >= 4) {
        max_exp = std::atoi(argv[3]);
        if (max_exp < 0) max_exp = 0;
    }

    std::string baseline = "PGM-mem-range";
    if (argc >= 5) {
        baseline = argv[4];
    }

    // 2) 文件路径
    std::string filename     = dataset_basename;
    std::string range_fname  = dataset_basename + ".range.bin";
    std::string file         = std::string(DATASETS) + filename;
    std::string range_file   = std::string(DATASETS) + range_fname;

    // 3) 加载数据 / ranges（整块载入内存）
    std::vector<KeyType> data          = load_data_pgm_safe<KeyType>(file, num_keys);
    std::vector<pgm::RangeQ> ranges    = load_ranges_pgm_safe(range_file);

    if (data.empty() || ranges.empty()) {
        std::cerr << "load_data or load_ranges failed, data=" << data.size()
                  << ", ranges=" << ranges.size() << "\n";
        return 1;
    }

    // 4) 构建内存 PGM 索引
    pgm::PGMIndex<KeyType, Epsilon, EpsilonRecur> index(data);

    // 5) 输出 CSV：<basename>_pgm_mem_range_multithread.csv
    std::string out_csv = dataset_basename + "_pgm_mem_range_multithread.csv";
    std::ofstream ofs(out_csv);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output csv: " << out_csv << "\n";
        return 1;
    }

    ofs << "baseline,threads,latency_ns,walltime_s,height,avg_IOs\n";

    // 6) 线程数从 1,2,4,...,2^max_exp
    const int RUNS = 3;
    for (int exp = 0; exp <= max_exp; exp++) {
        int threads = 1 << exp;

        double avg_latency = 0.0;
        double avg_wall    = 0.0;
        size_t avg_IOs     = 0;
        size_t height      = index.height();

        for (int r = 0; r < RUNS; r++) {
            BenchmarkResult br = run_experiment_range(data, ranges, index, threads);
            avg_latency += br.time_ns;
            avg_wall    += br.total_time_s;
            avg_IOs     += br.data_IOs;    // 永远是 0，这里只是占位
            height       = br.height;
        }
        avg_latency /= RUNS;
        avg_wall    /= RUNS;
        avg_IOs     /= RUNS;

        std::cout << "[PGM-mem-range] threads=" << threads
                  << ", avg_latency=" << avg_latency << " ns"
                  << ", wall=" << avg_wall << " s"
                  << ", IOs=" << avg_IOs
                  << ", height=" << height << std::endl;

        ofs << baseline << "," << threads << ","
            << avg_latency << ","
            << avg_wall    << ","
            << height      << ","
            << avg_IOs     << "\n";
    }

    ofs.close();
    return 0;
}
