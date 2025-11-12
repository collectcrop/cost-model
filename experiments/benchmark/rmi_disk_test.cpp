// g++ ./my_first_rmi.cpp ./test.cpp -o ./test -g
#include "my_first_rmi.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string>
#include <algorithm>
#include <time.h> 
#include "utils/utils.hpp"

using KeyType = uint64_t;
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"

struct BenchmarkResult {
    size_t epsilon;
    double time_ns;      // 平均 query latency (ns)
    double total_time_s; // wall time (秒)
    size_t data_IOs;
    size_t not_found;
    uint64_t index_cpu_ns;  // 新增：线程 CPU 时间累积
    uint64_t io_wait_ns;    // 新增：线程 IO 等待时间累积
};

using timer = std::chrono::high_resolution_clock;

struct ThreadArgs {
    std::vector<KeyType>* queries;
    size_t start;
    size_t end;
    BenchmarkResult result;
    std::vector<size_t> io_pages; 
};

// ========== settings ==========
const char* DATA_PATH   = "/mnt/home/zwshi/learned-index/cost-model/experiments/benchmark/rmi_data";   
const char* QUERY_FILE  = "/mnt/home/zwshi/Datasets/SOSD/books_10M_uint64_unique.query.bin";   
const char* DATA_FILE   = "/mnt/home/zwshi/Datasets/SOSD/books_10M_uint64_unique";                             

// ========== global ==========
std::atomic<size_t> IOs(0);
std::atomic<int> fd(-1);
// one-by-one
// void worker_thread(const std::vector<uint64_t>& queries, int fd) {
//     const size_t PAGE = 4096; // O_DIRECT 要求 4KB 对齐
//     void* aligned_buf = nullptr;

//     // 分配一个 4KB 对齐的缓冲区
//     if (posix_memalign(&aligned_buf, PAGE, PAGE) != 0) {
//         perror("posix_memalign");
//         return;
//     }

//     uint64_t* buf = reinterpret_cast<uint64_t*>(aligned_buf);
//     size_t keys_per_page = PAGE / sizeof(uint64_t);

//     for (auto key : queries) {
//         size_t err;
//         uint64_t pos = my_first_rmi::lookup(key, &err);

//         // 计算目标范围
//         uint64_t lo = (pos > err ? pos - err : 0);
//         uint64_t hi = pos + err;

//         // 转换成字节偏移，按 4KB 对齐
//         off_t offset = (lo * sizeof(uint64_t)) & ~(PAGE - 1);
//         off_t end    = ((hi + 1) * sizeof(uint64_t) + PAGE - 1) & ~(PAGE - 1);
//         size_t size  = end - offset;

//         // 循环读取，确保覆盖整个 [lo,hi]
//         for (off_t off = offset; off < end; off += PAGE) {
//             ssize_t n = pread(fd, buf, PAGE, off);
//             if (n < 0) {
//                 perror("pread");
//                 continue;
//             }

//             size_t count = n / sizeof(uint64_t);
//             // binary search
//             if (std::binary_search(buf, buf + count, key)) {
//                 break; // found
//             }
//         }
//     }

//     free(aligned_buf);
//     finished_queries += queries.size();
// }

// all-at-once
void* worker_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    const size_t PAGE = 4096; // O_DIRECT 要求 4KB 对齐
    size_t keys_per_page = PAGE / sizeof(uint64_t);

    timespec c0{}; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c0);
    auto t0 = timer::now();
    for (size_t i = args->start; i < args->end; i++){
        size_t err;
        auto q = (*(args->queries))[i];
        uint64_t pos = my_first_rmi::lookup(q, &err);

        uint64_t lo = (pos > err ? pos - err : 0);
        uint64_t hi = pos + err;

        off_t offset = (lo * sizeof(uint64_t)) & ~(PAGE - 1);
        off_t end    = ((hi + 1) * sizeof(uint64_t) + PAGE - 1) & ~(PAGE - 1);
        size_t size  = end - offset;
        void* aligned_buf = nullptr;

        if (posix_memalign(&aligned_buf, PAGE, size) != 0) {
            perror("posix_memalign");
            return nullptr;
        }

        uint64_t* buf = reinterpret_cast<uint64_t*>(aligned_buf);
        ssize_t n = pread(fd, buf, PAGE, offset);
        if (n < 0) {
            perror("pread");
            continue;
        }
        size_t count = n / sizeof(uint64_t);
        // binary search
        bool res = std::binary_search(buf, buf + count, q);
        if (!res) {
            // std::cout << "not found: " << q << std::endl;
            args->result.not_found++;
        }
        free(aligned_buf);
        // IOs += (size)/PAGE;
        IOs += 1;    
        args->io_pages.push_back(size);
    }
    auto t1 = timer::now();
    timespec c1{}; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c1);
    auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    uint64_t cpu_ns = (uint64_t)( (c1.tv_sec - c0.tv_sec) * 1000000000LL + (c1.tv_nsec - c0.tv_nsec) );
    uint64_t io_wait = (t > (long long)cpu_ns) ? (uint64_t)t - cpu_ns : 0ULL;

    args->result.time_ns = (double)t / (args->end - args->start);
    args->result.total_time_s = t / 1e9;
    args->result.data_IOs = IOs;
    args->result.index_cpu_ns = cpu_ns;
    args->result.io_wait_ns   = io_wait;
    return nullptr;
}

BenchmarkResult run_experiment(std::vector<KeyType>& queries,
                               int THREADS) {
    size_t total_queries = queries.size();
    size_t per_thread = total_queries / THREADS;

    pthread_t threads[THREADS];
    ThreadArgs args[THREADS];

    for (int i = 0; i < THREADS; i++) {
        args[i].queries = &queries;
        args[i].start = i * per_thread;
        args[i].end = (i == THREADS - 1) ? total_queries : (i + 1) * per_thread;
        pthread_create(&threads[i], nullptr, worker_thread, &args[i]);
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
    r.data_IOs = 0;
    r.time_ns = 0;
    r.index_cpu_ns = 0;
    r.io_wait_ns   = 0;

    for (int i = 0; i < THREADS; i++) {
        r.time_ns += args[i].result.time_ns;
        r.data_IOs += args[i].result.data_IOs;
        r.index_cpu_ns += args[i].result.index_cpu_ns;
        r.io_wait_ns   += args[i].result.io_wait_ns;
    }
    r.time_ns /= THREADS;

    // === 输出 CDF（每次 run_experiment 都会生成/覆盖）===
    std::vector<size_t> all_pages;
    all_pages.reserve(queries.size());
    for (int i = 0; i < THREADS; i++) {
        all_pages.insert(all_pages.end(), args[i].io_pages.begin(), args[i].io_pages.end());
    }
    std::sort(all_pages.begin(), all_pages.end());
    std::ofstream cdf_ofs("rmi_query_io_cdf.csv");
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

    // open data file
    fd = open(DATA_FILE, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open data file");
        return 1;
    }
    // construct index
    if (!my_first_rmi::load(DATA_PATH)) {
        std::cerr << "Failed to load RMI model!" << std::endl;
        return 1;
    }

    std::ofstream ofs("rmi_disk_multithread.csv");
    ofs << "threads,avg_latency_ns,avg_walltime_s,avg_IOs,index_cpu_ns,io_wait_ns\n";

    for (int exp = 0; exp <= 14; exp++) {
        int threads = 1 << exp;
        double avg_latency = 0;
        double avg_wall = 0;
        size_t avg_IOs = 0;
        size_t height = 0;

        for (int t = 0; t < 3; t++) {
            BenchmarkResult r = run_experiment(queries, threads);
            avg_latency = r.time_ns;
            avg_wall = r.total_time_s;
            avg_IOs = r.data_IOs;
            std::cout << "threads=" << threads
                    << ", not found=" << r.not_found
                    << ", avg_latency=" << avg_latency << " ns"
                    << ", wall=" << avg_wall << " s"
                    << ", IOs=" << avg_IOs 
                    << ", Index CPU Time=" << r.index_cpu_ns 
                    << ", IO Time=" << r.io_wait_ns << std::endl;

            ofs << threads << "," << avg_latency << "," << avg_wall
                << "," << avg_IOs << "," << r.index_cpu_ns << "," << r.io_wait_ns << "\n";
        }
    }

    ofs.close();
    return 0;
}
