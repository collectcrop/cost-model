#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

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

struct BenchmarkResult {
    size_t epsilon;
    double time_ns;      // 平均 query latency (ns)
    double total_time_s; // wall time (秒)
    size_t data_IOs;
    size_t height;
};

using timer = std::chrono::high_resolution_clock;

struct ThreadArgs {
    std::vector<KeyType>* queries;
    pgm::PGMIndexDisk<KeyType, 8>* index;
    size_t start;
    size_t end;
    BenchmarkResult result;
};

void* query_worker(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;

    auto t0 = timer::now();
    for (size_t i = args->start; i < args->end; i++) {
        auto q = (*(args->queries))[i];
        auto range = args->index->search(q, pgm::ALL_IN_ONCE);
        std::vector<pgm::Record> records = range.records;
        size_t lo = range.lo;
        size_t hi = range.hi;
        binary_search_record(records.data(), lo, hi, q);
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
                               pgm::PGMIndexDisk<KeyType, 8>& index,
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
    r.epsilon = 8;
    r.total_time_s = total_ns / 1e9;
    r.height = index.height();
    r.data_IOs = 0;
    r.time_ns = 0;

    for (int i = 0; i < THREADS; i++) {
        r.time_ns += args[i].result.time_ns;
        r.data_IOs += args[i].result.data_IOs;
    }
    r.time_ns /= THREADS;
    return r;
}

int main() {
    std::string filename = "books_10M_uint64_unique";
    std::string query_filename = "books_10M_uint64_unique.query.bin";
    std::string file = DATASETS + filename;
    std::string query_file = DATASETS + query_filename;

    std::vector<KeyType> data = load_data(file, 10000000);
    std::vector<KeyType> queries = load_queries(query_file);

    // 构建一次索引，确保文件存在
    pgm::PGMIndexDisk<KeyType, 8> index(data, file);

    std::ofstream ofs("pgm_disk_multithread.csv");
    ofs << "threads,avg_latency_ns,avg_walltime_s,height,avg_IOs\n";

    for (int exp = 0; exp <= 10; exp++) {
        int threads = 1 << exp;
        double avg_latency = 0;
        double avg_wall = 0;
        size_t avg_IOs = 0;
        size_t height = 0;

        for (int t = 0; t < 5; t++) {
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
