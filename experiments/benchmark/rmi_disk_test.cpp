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
#include "utils/utils.hpp"

using KeyType = uint64_t;
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"

struct BenchmarkResult {
    size_t epsilon;
    double time_ns;      // 平均 query latency (ns)
    double total_time_s; // wall time (秒)
    size_t data_IOs;
    size_t not_found;
};

using timer = std::chrono::high_resolution_clock;

struct ThreadArgs {
    std::vector<KeyType>* queries;
    size_t start;
    size_t end;
    BenchmarkResult result;
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
    }
    auto t1 = timer::now();
    auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    args->result.time_ns = (double)t / (args->end - args->start);
    args->result.total_time_s = t / 1e9;
    args->result.data_IOs = IOs;
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
    ofs << "threads,avg_latency_ns,avg_walltime_s,avg_IOs\n";

    for (int exp = 4; exp <= 6; exp++) {
        int threads = 1 << exp;
        double avg_latency = 0;
        double avg_wall = 0;
        size_t avg_IOs = 0;
        size_t height = 0;

        for (int t = 0; t < 5; t++) {
            BenchmarkResult r = run_experiment(queries, threads);
            avg_latency = r.time_ns;
            avg_wall = r.total_time_s;
            avg_IOs = r.data_IOs;
            std::cout << "threads=" << threads
                    << ", not found=" << r.not_found
                    << ", avg_latency=" << avg_latency << " ns"
                    << ", wall=" << avg_wall << " s"
                    << ", IOs=" << avg_IOs << std::endl;

            ofs << threads << "," << avg_latency << "," << avg_wall
                << "," << avg_IOs << "\n";
        }
    }

    ofs.close();
    return 0;
}
// int main() {
//     // 1. 加载 RMI 模型
//     if (!my_first_rmi::load(DATA_PATH)) {
//         std::cerr << "Failed to load RMI model!" << std::endl;
//         return 1;
//     }

//     // 2. 打开数据文件
//     int fd = open(DATA_FILE, O_RDONLY | O_DIRECT);
//     if (fd < 0) {
//         perror("open data file");
//         return 1;
//     }

//     // 3. 读取 queries.bin
//     std::ifstream qf(QUERY_FILE, std::ios::binary);
//     if (!qf) {
//         std::cerr << "Failed to open query file!" << std::endl;
//         return 1;
//     }
//     qf.seekg(0, std::ios::end);
//     size_t fsize = qf.tellg();
//     qf.seekg(0, std::ios::beg);

//     size_t num_queries = fsize / sizeof(uint64_t);
//     std::vector<uint64_t> all_queries(num_queries);
//     qf.read(reinterpret_cast<char*>(all_queries.data()), fsize);
//     qf.close();

//     std::cout << "Loaded " << num_queries << " queries" << std::endl;

//     // 4. 切分 query 到各线程
//     std::vector<std::thread> threads;
//     size_t chunk = num_queries / NUM_THREADS;

//     auto start = std::chrono::high_resolution_clock::now();
//     for (int t = 0; t < NUM_THREADS; t++) {
//         size_t begin = t * chunk;
//         size_t end   = (t == NUM_THREADS - 1 ? num_queries : begin + chunk);
//         std::vector<uint64_t> q(all_queries.begin()+begin, all_queries.begin()+end);
//         threads.emplace_back(worker_thread, q, fd);
//     }

//     for (auto& th : threads) th.join();
//     auto end = std::chrono::high_resolution_clock::now();

//     double seconds = std::chrono::duration<double>(end - start).count();
//     double iops = IOs / seconds;

//     std::cout << "IOs: " << IOs << std::endl;
//     std::cout << "Time: " << seconds << " s" << std::endl;
//     std::cout << "IOPS: " << iops << std::endl;

//     // 5. 清理
//     close(fd);
//     my_first_rmi::cleanup();
//     return 0;
// }
