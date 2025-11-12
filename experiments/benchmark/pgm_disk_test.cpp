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
#include "pgm/pgm_index_disk.hpp"
#include "utils/utils.hpp"


using KeyType = uint64_t;
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"
#define Epsilon 76
#define EpsilonRecur 32

struct BenchmarkResult {
    size_t epsilon;
    double time_ns;      // 平均 query latency (ns)
    double total_time_s; // wall time (秒)
    size_t data_IOs;
    size_t height;
    uint64_t index_cpu_ns;  
    uint64_t io_wait_ns; 
};

using timer = std::chrono::high_resolution_clock;

struct ThreadArgs {
    std::vector<KeyType>* queries;
    pgm::PGMIndexDisk<KeyType, Epsilon, EpsilonRecur>* index;
    size_t start;
    size_t end;
    BenchmarkResult result;
    std::vector<size_t> io_pages; 
};

void* query_worker(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    const size_t PAGE = 4096;
    auto t0 = timer::now();
    for (size_t i = args->start; i < args->end; i++) {
        auto q = (*(args->queries))[i];
        auto range = args->index->search(q, pgm::ALL_IN_ONCE);
        std::vector<pgm::Record> records = range.records;
        size_t lo = range.lo;
        size_t hi = range.hi;
        binary_search_record(records.data(), lo, hi, q);
        // 基于 lo/hi 计算该 query 的页级 I/O 量
        off_t offset = (lo * sizeof(KeyType)) & ~(PAGE - 1);
        off_t end    = ((hi + 1) * sizeof(KeyType) + PAGE - 1) & ~(PAGE - 1);
        size_t size  = end - offset;
        size_t pages = size / PAGE;
        if (pages == 0) pages = 1;
        args->io_pages.push_back(size);
    }
    auto t1 = timer::now();

    auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    args->result.time_ns = (double)t / (args->end - args->start);
    args->result.total_time_s = t / 1e9;
    args->result.data_IOs = args->index->get_IOs();
    args->result.height = args->index->height();

    return nullptr;
}

BenchmarkResult run_experiment(std::vector<KeyType>& queries,
                               pgm::PGMIndexDisk<KeyType, Epsilon, EpsilonRecur>& index,
                               int THREADS) {
    size_t total_queries = queries.size();
    size_t per_thread = total_queries / THREADS;

    pthread_t threads[THREADS];
    ThreadArgs args[THREADS];

    for (int i = 0; i < THREADS; i++) {
        args[i].queries = &queries;
        args[i].start = i * per_thread;
        args[i].end = (i == THREADS - 1) ? total_queries : (i + 1) * per_thread;
        args[i].index = &index;
        pthread_create(&threads[i], nullptr, query_worker, &args[i]);
    }

    auto t0 = timer::now();
    for (int i = 0; i < THREADS; i++) {
        pthread_join(threads[i], nullptr);
    }
    auto t1 = timer::now();

    auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // 汇总
    BenchmarkResult r{};
    r.epsilon = Epsilon;
    r.total_time_s = total_ns / 1e9;
    r.height = index.height();
    r.data_IOs = 0;
    r.time_ns = 0;

    for (int i = 0; i < THREADS; i++) {
        r.time_ns += args[i].result.time_ns;
        r.data_IOs += args[i].result.data_IOs;
    }
    r.time_ns /= THREADS;

    // === 输出 CDF（每次 run_experiment 都会生成/覆盖）===
    std::vector<size_t> all_pages;
    all_pages.reserve(queries.size());
    for (int i = 0; i < THREADS; i++) {
        all_pages.insert(all_pages.end(), args[i].io_pages.begin(), args[i].io_pages.end());
    }
    std::sort(all_pages.begin(), all_pages.end());
    std::ofstream cdf_ofs("pgm_query_io_cdf.csv");
    cdf_ofs << "io_size,cdf\n";
    const size_t N = all_pages.size();
    for (size_t i = 0; i < N; ) {
        size_t v = all_pages[i];
        size_t j = i + 1;
        while (j < N && all_pages[j] == v) ++j;
        double cdf = static_cast<double>(j) / static_cast<double>(N);
        cdf_ofs << v << "," << cdf << "\n";
        i = j;
    }
    cdf_ofs.close();
    return r;
}

int main() {
    std::string filename = "osm_cellids_10M_uint64_unique";
    std::string query_filename = "osm_cellids_10M_uint64_unique.query.bin";
    std::string file = DATASETS + filename;
    std::string query_file = DATASETS + query_filename;

    std::vector<KeyType> data = load_data(file, 10000000);
    std::vector<KeyType> queries = load_queries(query_file);

    // 构建一次索引，确保文件存在
    pgm::PGMIndexDisk<KeyType, Epsilon, EpsilonRecur> index(data, file);

    std::ofstream ofs("pgm_disk_multithread.csv");
    ofs << "threads,avg_latency_ns,avg_walltime_s,height,avg_IOs\n";

    for (int exp = 0; exp <= 14; exp++) {
        int threads = 1 << exp;
        double avg_latency = 0;
        double avg_wall = 0;
        size_t avg_IOs = 0;
        size_t height = 0;

        for (int t = 0; t < 3; t++) {
            BenchmarkResult r = run_experiment(queries, index, threads);
            avg_latency = r.time_ns;
            avg_wall = r.total_time_s;
            avg_IOs = r.data_IOs;
            height = r.height;
            std::cout << "threads=" << threads
                    << ", avg_latency=" << avg_latency << " ns"
                    << ", wall=" << avg_wall << " s"
                    << ", IOs=" << avg_IOs
                    << ", height=" << height << std::endl;

            ofs << threads << "," << avg_latency << "," << avg_wall
                << "," << height << "," << avg_IOs << "\n";
        }
    }

    ofs.close();
    return 0;
}
