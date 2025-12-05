#include "rmi/books_rmi.h"
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
#include <pthread.h>
#include <iomanip>

#include "utils/utils.hpp"   // 里头需要有 load_queries

using KeyType = uint64_t;
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"

// RMI 模型文件所在目录（和你原来 DATA_PATH 一样）
const char* RMI_DATA_PATH = "/mnt/home/zwshi/learned-index/cost-model/include/rmi/rmi_data";

// 方便改成别的 RMI 命名空间时统一修改
namespace RMI = books_rmi;

struct BenchmarkResult {
    size_t   epsilon;
    double   time_ns;        // 平均 query latency (ns)
    double   total_time_s;   // 整体 wall time (s)
    size_t   data_IOs;       // 总 IO 次数（pread 次数）
    size_t   not_found;      // 未命中数（对 point query 来说）
    uint64_t index_cpu_ns;   // 线程 CPU 时间累积 (ns)
    uint64_t io_wait_ns;     // IO 等待时间累积 (ns)
};

using timer = std::chrono::high_resolution_clock;

// ========== global ==========
std::atomic<int> fd(-1);
size_t GLOBAL_N_KEYS = 0;   // 用于边界 clamp

struct ThreadArgs {
    std::vector<KeyType>* queries;
    size_t start;
    size_t end;
    BenchmarkResult result;
    std::vector<size_t> io_pages; // 如果要做 CDF 可以用
};

// 单线程 worker：处理 [start, end) 的 queries
void* worker_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    const size_t PAGE = 4096;

    timespec c0{}; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c0);
    auto t_thread_start = timer::now();

    size_t local_IOs   = 0;
    size_t not_found   = 0;
    uint64_t local_lat_ns = 0;  // ★ 每个 query 独立计时后累加

    for (size_t i = args->start; i < args->end; i++) {
        auto q = (*(args->queries))[i];

        auto q_t0 = timer::now();

        size_t err = 0;
        uint64_t pos = RMI::lookup(q, &err);

        uint64_t lo_idx = (pos > err ? pos - err : 0);
        uint64_t hi_idx = pos + err;
        if (hi_idx >= GLOBAL_N_KEYS) hi_idx = GLOBAL_N_KEYS - 1;

        off_t byte_lo = static_cast<off_t>(lo_idx) * sizeof(uint64_t);
        off_t byte_hi = static_cast<off_t>(hi_idx + 1) * sizeof(uint64_t);

        off_t offset = byte_lo & ~(PAGE - 1);
        off_t end    = (byte_hi + PAGE - 1) & ~(PAGE - 1);
        size_t size  = static_cast<size_t>(end - offset);

        void* aligned_buf = nullptr;
        if (posix_memalign(&aligned_buf, PAGE, size) != 0) {
            perror("posix_memalign");
            continue;
        }
        uint64_t* buf = reinterpret_cast<uint64_t*>(aligned_buf);

        ssize_t n = pread(fd.load(), buf, size, offset);
        if (n < 0) {
            perror("pread");
            free(aligned_buf);
            continue;
        }

        size_t count = static_cast<size_t>(n) / sizeof(uint64_t);

        // 在 buffer 中二分查找 q
        bool res = std::binary_search(buf, buf + count, q);
        if (!res) not_found++;

        local_IOs += 1;
        args->io_pages.push_back(size);

        free(aligned_buf);

        auto q_t1 = timer::now();
        local_lat_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(q_t1 - q_t0).count();
    }

    auto t_thread_end = timer::now();
    timespec c1{}; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c1);

    auto thread_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_thread_end - t_thread_start).count();
    uint64_t cpu_ns = (uint64_t)((c1.tv_sec - c0.tv_sec) * 1000000000LL +
                                 (c1.tv_nsec - c0.tv_nsec));
    uint64_t io_wait = (thread_ns > (long long)cpu_ns) ? (uint64_t)thread_ns - cpu_ns : 0ULL;

    size_t qcnt = args->end - args->start;
    double avg_q_latency = qcnt ? (double)local_lat_ns / (double)qcnt : 0.0;

    args->result.epsilon      = 0;       // RMI 没有显式 epsilon，这里填 0 或者你的误差 bound
    args->result.time_ns      = avg_q_latency;
    args->result.total_time_s = thread_ns / 1e9;
    args->result.data_IOs     = local_IOs;
    args->result.index_cpu_ns = cpu_ns;
    args->result.io_wait_ns   = io_wait;
    args->result.not_found    = not_found;

    return nullptr;
}

// 运行一次实验：给定线程数 threads，返回聚合后的 BenchmarkResult
BenchmarkResult run_experiment(std::vector<KeyType>& queries,
                               int THREADS) {
    size_t total_queries = queries.size();
    size_t per_thread = total_queries / THREADS;

    std::vector<pthread_t> threads(THREADS);
    std::vector<ThreadArgs> args(THREADS);

    for (int i = 0; i < THREADS; i++) {
        args[i].queries = &queries;
        args[i].start   = i * per_thread;
        args[i].end     = (i == THREADS - 1) ? total_queries : (i + 1) * per_thread;
        pthread_create(&threads[i], nullptr, worker_thread, &args[i]);
    }

    auto t0 = timer::now();
    for (int i = 0; i < THREADS; i++) {
        pthread_join(threads[i], nullptr);
    }
    auto t1 = timer::now();

    auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    BenchmarkResult r{};
    r.epsilon       = 0;
    r.total_time_s  = total_ns / 1e9;
    r.data_IOs      = 0;
    r.time_ns       = 0;
    r.index_cpu_ns  = 0;
    r.io_wait_ns    = 0;
    r.not_found     = 0;

    // 真正的 per-query 平均 latency：对每个线程的 avg_latency 做 weighted average
    uint64_t sum_lat_ns = 0;

    for (int i = 0; i < THREADS; i++) {
        size_t qcnt = args[i].end - args[i].start;
        double avg_lat_this_thread = args[i].result.time_ns; // 每个 query 的平均 latency
        sum_lat_ns += (uint64_t)(avg_lat_this_thread * (double)qcnt);

        r.data_IOs      += args[i].result.data_IOs;
        r.index_cpu_ns  += args[i].result.index_cpu_ns;
        r.io_wait_ns    += args[i].result.io_wait_ns;
        r.not_found     += args[i].result.not_found;
    }

    if (total_queries > 0) {
        r.time_ns = (double)sum_lat_ns / (double)total_queries;
    } else {
        r.time_ns = 0.0;
    }

    return r;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " <dataset_basename> <num_keys> [max_log2_threads] [repeats]\n\n"
                  << "Example:\n  " << argv[0]
                  << " books_200M_uint64_unique 200000000 10 3\n";
        return 1;
    }

    std::string dataset_basename = argv[1];   // e.g. books_200M_uint64_unique
    GLOBAL_N_KEYS = std::strtoull(argv[2], nullptr, 10);

    int max_exp = 10;
    if (argc >= 4) {
        max_exp = std::atoi(argv[3]);
        if (max_exp < 0) max_exp = 0;
    }

    int repeats = 3;
    if (argc >= 5) {
        repeats = std::atoi(argv[4]);
        if (repeats <= 0) repeats = 1;
    }

    std::string data_file  = std::string(DATASETS) + dataset_basename;
    std::string query_file = std::string(DATASETS) + dataset_basename + ".query.bin";

    // 如果你只是为了保证文件存在，这个 load_data 可以删掉
    // std::vector<KeyType> data = load_data(data_file, GLOBAL_N_KEYS);

    std::vector<KeyType> queries = load_queries(query_file);
    if (queries.empty()) {
        std::cerr << "Failed to load queries from " << query_file << "\n";
        return 1;
    }
    std::cout << "Loaded " << queries.size() << " queries from " << query_file << "\n";

    // open data file (O_DIRECT)
    fd = ::open(data_file.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open data file");
        return 1;
    }

    // load RMI model
    if (!RMI::load(RMI_DATA_PATH)) {
        std::cerr << "Failed to load RMI model from " << RMI_DATA_PATH << "!\n";
        return 1;
    }

    // 输出 CSV：<dataset_basename>_rmi_disk_range.csv
    std::string csv_name = dataset_basename + "_rmi_disk_range.csv";
    std::ofstream ofs(csv_name);
    if (!ofs) {
        std::cerr << "Failed to open CSV file: " << csv_name << "\n";
        return 1;
    }

    ofs << "baseline,threads,latency_ns,walltime_s,avg_IOs,index_cpu_ns,io_wait_ns\n";
    ofs << std::fixed << std::setprecision(6);

    for (int exp = 0; exp <= max_exp; exp++) {
        int threads = 1 << exp;
        if (threads > (int)queries.size()) {
            threads = (int)queries.size();
            if (threads == 0) threads = 1;
        }

        double avg_latency_ns = 0.0;
        double avg_wall_s     = 0.0;
        double avg_IOs        = 0.0;
        uint64_t avg_cpu_ns   = 0;
        uint64_t avg_io_wait  = 0;
        size_t avg_not_found  = 0;

        for (int r = 0; r < repeats; ++r) {
            BenchmarkResult br = run_experiment(queries, threads);

            avg_latency_ns += br.time_ns;
            avg_wall_s     += br.total_time_s;
            avg_IOs        += (double)br.data_IOs;
            avg_cpu_ns     += br.index_cpu_ns;
            avg_io_wait    += br.io_wait_ns;
            avg_not_found  += br.not_found;
        }

        avg_latency_ns /= repeats;
        avg_wall_s     /= repeats;
        avg_IOs        /= repeats;
        avg_cpu_ns     /= repeats;
        avg_io_wait    /= repeats;
        avg_not_found  /= repeats;

        std::cout << "threads=" << threads
                  << ", not_found=" << avg_not_found
                  << ", avg_latency=" << avg_latency_ns << " ns"
                  << ", wall=" << avg_wall_s << " s"
                  << ", IOs=" << avg_IOs
                  << ", Index CPU Time=" << avg_cpu_ns
                  << ", IO Time=" << avg_io_wait
                  << std::endl;

        ofs << "RMI-disk," << threads << ","
            << avg_latency_ns << ","
            << avg_wall_s << ","
            << avg_IOs << ","
            << avg_cpu_ns << ","
            << avg_io_wait << "\n";
        ofs.flush();
    }

    ofs.close();
    std::cout << "Done. Results written to " << csv_name << "\n";

    RMI::cleanup();
    ::close(fd.load());
    return 0;
}
