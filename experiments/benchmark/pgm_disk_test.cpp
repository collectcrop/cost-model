#include <cstdlib>
#include <sstream> 
#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h> 
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>
#include "distribution/zipf.hpp"
#include "pgm/pgm_index.hpp"
#include "utils/utils.hpp"

struct ApproxPosExt {
    std::vector<pgm::Record> records;  // record buffer
    size_t lo;      // minimal pos in buffer
    size_t hi;      // maximal pos in buffer
};

using KeyType = uint64_t;
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"
#define Epsilon 16
#define EpsilonRecur 4
#define use_direct true

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
    int fd;  
    pgm::PGMIndex<KeyType, Epsilon, EpsilonRecur>* index;
    size_t start;
    size_t end;
    BenchmarkResult result;
    std::vector<size_t> io_pages; 
};

ApproxPosExt search(size_t lo, size_t hi, int data_fd, int &IOs){
    std::vector<pgm::Record> buffer;
    
    // std::vector<pgm::Page> pages = cache->get(lo / ITEM_PER_PAGE, hi / ITEM_PER_PAGE);
    std::vector<pgm::Page> pages;
    size_t offset = lo / pgm::ITEM_PER_PAGE;
    size_t len = hi / pgm::ITEM_PER_PAGE - offset + 1;
    void* raw = nullptr;
    if (posix_memalign(&raw, pgm::PAGE_SIZE, pgm::PAGE_SIZE*len) != 0) {
        throw std::runtime_error("posix_memalign failed");
    }
    std::shared_ptr<char[]> buf(reinterpret_cast<char*>(raw), [](char* p){ free(p); });
    ssize_t br = pread(data_fd, buf.get(), pgm::PAGE_SIZE*len, offset*pgm::PAGE_SIZE);
    if (br < 0){
        throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
    }
    size_t pages_read = (br + pgm::PAGE_SIZE - 1) / pgm::PAGE_SIZE;
    IOs += pages_read;
    for (size_t i = 0; i < pages_read; i++) {
        pgm::Page page;
        void* page_ptr = nullptr;
        if (posix_memalign(&page_ptr, pgm::PAGE_SIZE, pgm::PAGE_SIZE) != 0) {
            throw std::runtime_error("posix_memalign failed for sub-page");
        }
        page.data.reset(reinterpret_cast<char*>(page_ptr), [](char* p){ free(p); });

        size_t copy_size = std::min(static_cast<size_t>(br - i * pgm::PAGE_SIZE), (size_t)pgm::PAGE_SIZE);
        memcpy(page.data.get(), buf.get() + i * pgm::PAGE_SIZE, copy_size);

        page.valid_len = copy_size;
        pages.push_back(std::move(page));
    }

    for (auto &page : pages) {
        size_t num_records = page.valid_len / sizeof(pgm::Record);   // valid record num
        pgm::Record* records = reinterpret_cast<pgm::Record*>(page.data.get());
        buffer.insert(buffer.end(), records, records + num_records);
    }

    size_t page_lo = lo / pgm::ITEM_PER_PAGE;
    size_t rel_lo = lo % pgm::ITEM_PER_PAGE;
    size_t rel_hi = std::min(hi - page_lo * pgm::ITEM_PER_PAGE, buffer.size());
    
    return {std::move(buffer), rel_lo, rel_hi};
}

void* query_worker(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    const size_t PAGE = 4096;
    int IOs = 0;
    timespec c0{}; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c0);
    auto t0 = timer::now();
    for (size_t i = args->start; i < args->end; i++) {
        auto q = (*(args->queries))[i];
        auto range = args->index->search(q);
        // std::vector<pgm::Record> records = range.records;
        size_t lo = range.lo;
        size_t hi = range.hi;
        std::vector<pgm::Record> records = search(lo, hi, args->fd, IOs).records;
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
    timespec c1{}; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c1);
    auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    uint64_t cpu_ns = (uint64_t)((c1.tv_sec - c0.tv_sec) * 1000000000LL +
                                 (c1.tv_nsec - c0.tv_nsec));
    uint64_t io_wait = (t > (long long)cpu_ns) ? (uint64_t)t - cpu_ns : 0ULL;

    args->result.time_ns = (double)t / (args->end - args->start);
    args->result.total_time_s = t / 1e9;
    args->result.data_IOs = IOs;
    args->result.height = args->index->height();
    args->result.index_cpu_ns = cpu_ns;
    args->result.io_wait_ns   = io_wait;

    return nullptr;
}

BenchmarkResult run_experiment(std::vector<KeyType>& queries,
                               pgm::PGMIndex<KeyType, Epsilon, EpsilonRecur>& index,
                               int data_fd,
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
        args[i].fd = data_fd;
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
    r.index_cpu_ns  = 0;
    r.io_wait_ns    = 0;

    for (int i = 0; i < THREADS; i++) {
        r.time_ns += args[i].result.time_ns;
        r.data_IOs += args[i].result.data_IOs;
        r.index_cpu_ns  += args[i].result.index_cpu_ns;
        r.io_wait_ns    += args[i].result.io_wait_ns;
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

int main(int argc, char** argv) {
    // helper
    if (argc < 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " <dataset_basename> <num_keys> [max_log2_threads] [baseline_name]\n\n"
                  << "Example:\n  " << argv[0]
                  << " books_200M_uint64_unique 200000000 10 PGM-disk\n";
        return 1;
    }

    // 1) essential parameter
    std::string dataset_basename = argv[1];              // 如 books_200M_uint64_unique
    uint64_t num_keys = std::strtoull(argv[2], nullptr, 10);  // 如 200000000

    // 2) optional parameter
    int max_exp = 10;  // 默认 2^0 .. 2^10
    if (argc >= 4) {
        max_exp = std::atoi(argv[3]);
        if (max_exp < 0) max_exp = 0;
    }

    // 3) 可选参数：baseline 名字
    std::string baseline = "PGM-disk";
    if (argc >= 5) {
        baseline = argv[4];
    }

    // 4) 拼接文件路径
    std::string filename       = dataset_basename;                // books_200M_uint64_unique
    std::string query_filename = dataset_basename + ".query.bin"; // books_200M_uint64_unique.query.bin
    std::string file       = DATASETS + filename;
    std::string query_file = DATASETS + query_filename;

    // 5) 加载数据
    std::vector<KeyType> data    = load_data_pgm_safe<KeyType>(file, 200000000);
    std::vector<KeyType> queries = load_queries_pgm_safe<KeyType>(query_file);
    
    // 6) 构建索引
    pgm::PGMIndex<KeyType, Epsilon, EpsilonRecur> index(data);

    // 7) 打开数据文件
    int data_fd;
    if (use_direct){ data_fd = open(file.c_str(), O_RDONLY | O_DIRECT); }
    else { data_fd = open(file.c_str(), O_RDONLY);}
    if (data_fd < 0) {
        perror("open data file");
        return 1;
    }

    // 8) 输出文件名：<basename>_pgm_disk_multithread.csv
    std::string out_csv = dataset_basename + "_pgm_disk_multithread.csv";
    std::ofstream ofs(out_csv);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output csv: " << out_csv << "\n";
        close(data_fd);
        return 1;
    }

    ofs << "baseline,threads,latency_ns,walltime_s,height,avg_IOs,mem_time_s,io_time_s\n";

    // 9) 线程实验循环
    for (int exp = 0; exp <= max_exp; exp++) {
        int threads = 1 << exp;
        double avg_latency = 0;
        double avg_wall    = 0;
        size_t avg_IOs     = 0;
        size_t height      = 0;

        for (int t = 0; t < 3; t++) {
            BenchmarkResult r = run_experiment(queries, index, data_fd, threads);
            avg_latency = r.time_ns;
            avg_wall    = r.total_time_s;
            avg_IOs     = r.data_IOs;
            height      = r.height;
            double mem_time_s = r.index_cpu_ns / 1e9;
            double io_time_s  = r.io_wait_ns   / 1e9;
            std::cout << "threads=" << threads
                      << ", avg_latency=" << avg_latency << " ns"
                      << ", wall=" << avg_wall << " s"
                      << ", IOs=" << avg_IOs
                      << ", height=" << height 
                      << ", mem_time_s=" << mem_time_s
                      << ", io_time_s=" << io_time_s << std::endl;

            ofs << baseline << "," << threads << "," << avg_latency << "," << avg_wall
                << "," << height << "," << avg_IOs 
                << "," << mem_time_s << "," << io_time_s <<"\n";
        }
    }

    ofs.close();
    close(data_fd);
    return 0;
}
