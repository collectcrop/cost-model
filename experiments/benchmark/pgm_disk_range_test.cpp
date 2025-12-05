// pgm_disk_range_parallel.cpp
#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <thread>
#include <atomic>
#include <fcntl.h>
#include <iomanip>
#include <stdexcept>

#include "pgm/pgm_index.hpp"
#include "utils/include.hpp"   // KeyType, pgm::Record, PAGE_SIZE, etc.
#include "utils/utils.hpp"     // load_data_pgm_safe, load_ranges_pgm_safe

using KeyType = uint64_t;
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"

struct BenchmarkResult {
    size_t   epsilon;
    double   avg_latency_ns;   // 每个 query 独立计时后的平均值
    double   hit_ratio;        // PGM-disk 无缓存，这里恒 0
    long long wall_ns;         // 整体 wall time（方便算吞吐）
    long long data_io_ns;      // 所有 pread 的时间总和 (ns)
    size_t   height;           // PGM height
    uint64_t physical_ios;     // 实际发起的 IO 次数（一段连续 pread 算一次）
    uint64_t logical_ios;      // 逻辑访问 page 数（总共读了多少页）
};

// ---- worker：对 [begin, end) 范围内的 queries 做 PGM-disk range 查询 ----
template <size_t Epsilon>
static void worker_range_pgm_disk(const pgm::PGMIndex<KeyType, Epsilon>& index,
                                  int fd,
                                  const std::vector<pgm::RangeQ>& queries,
                                  size_t begin,
                                  size_t end,
                                  std::atomic<uint64_t>& logical_ios,
                                  std::atomic<uint64_t>& physical_ios,
                                  std::atomic<long long>& io_time_ns,
                                  std::atomic<long long>& query_lat_ns_total)
{
    using Clock = std::chrono::high_resolution_clock;
    const size_t PAGE = pgm::PAGE_SIZE;

    if (begin >= end) {
        return;
    }

    // 1) 预扫一遍，找出该线程负责 queries 中最大的 page span，用于分配 buffer
    size_t max_span_pages = 1;
    for (size_t i = begin; i < end; ++i) {
        const auto& q = queries[i];
        auto [plo, phi] = index.estimate_pages_for_range(q.lo, q.hi);
        if (phi >= plo) {
            size_t span = phi - plo + 1;
            if (span > max_span_pages) {
                max_span_pages = span;
            }
        }
    }

    size_t buf_bytes = max_span_pages * PAGE;
    void* raw = nullptr;
    if (posix_memalign(&raw, PAGE, buf_bytes) != 0) {
        throw std::runtime_error("posix_memalign failed");
    }
    char* buf = reinterpret_cast<char*>(raw);

    // ★ 每个线程维护一个结果向量，循环复用，避免每个 query 都重新分配
    std::vector<KeyType> results;
    results.reserve(256); // 预留一点容量，具体大小无所谓，只是减少 realloc

    long long local_query_lat_ns = 0;

    // 2) 逐个 query 处理
    for (size_t i = begin; i < end; ++i) {
        const auto& q = queries[i];

        // 对单个 query 的总 latency 计时（PGM estimate + IO + in-buffer scan）
        auto q_t0 = Clock::now();

        auto [page_lo, page_hi] = index.estimate_pages_for_range(q.lo, q.hi);
        if (page_hi < page_lo) {
            auto q_t1 = Clock::now();
            local_query_lat_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(q_t1 - q_t0).count();
            continue;
        }

        size_t span_pages = page_hi - page_lo + 1;
        size_t read_bytes = span_pages * PAGE;
        off_t off = static_cast<off_t>(page_lo) * static_cast<off_t>(PAGE);

        // 一次性把 [page_lo, page_hi] 全部读进 buffer
        auto io_t0 = Clock::now();
        ssize_t br = pread(fd, buf, read_bytes, off);
        auto io_t1 = Clock::now();

        if (br > 0) {
            long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(io_t1 - io_t0).count();
            io_time_ns.fetch_add(ns, std::memory_order_relaxed);
            logical_ios.fetch_add(span_pages, std::memory_order_relaxed);
            physical_ios.fetch_add(1, std::memory_order_relaxed);

            auto* recs = reinterpret_cast<pgm::Record*>(buf);
            size_t cnt = static_cast<size_t>(br) / sizeof(pgm::Record);
            if (cnt > 0) {
                // 在整段 buffer 上做一次 lower_bound / upper_bound
                auto* lb = std::lower_bound(
                    recs, recs + cnt, q.lo,
                    [](const pgm::Record& a, uint64_t k) { return a.key < k; }
                );
                auto* ub = std::upper_bound(
                    recs, recs + cnt, q.hi,
                    [](uint64_t k, const pgm::Record& a) { return k < a.key; }
                );

                // ★ 真正把 [lo, hi] 内的 key 全部 push 到结果集里
                results.clear();
                for (auto* cur = lb; cur < ub; ++cur) {
                    results.push_back(cur->key);
                }
                (void)results.size();
            }
        }

        auto q_t1 = Clock::now();
        local_query_lat_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(q_t1 - q_t0).count();
    }

    // 把该线程累加的 query latency 一次性写回
    query_lat_ns_total.fetch_add(local_query_lat_ns, std::memory_order_relaxed);

    free(raw);
}

// ---- PGM-disk range benchmark driver ----
template <size_t Epsilon>
static BenchmarkResult bench_range_pgm_disk(const std::vector<KeyType>& data,
                                            const std::vector<pgm::RangeQ>& ranges,
                                            const std::string& datafile,
                                            int num_threads)
{
    using Clock = std::chrono::high_resolution_clock;

    // 1) 构建 PGM 索引（纯内存）
    pgm::PGMIndex<KeyType, Epsilon> index(data);

    // 2) 打开数据文件（O_DIRECT）
    int fd = ::open(datafile.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open data");
        std::exit(1);
    }

    std::atomic<uint64_t> logical_ios{0};
    std::atomic<uint64_t> physical_ios{0};
    std::atomic<long long> io_time_ns{0};
    std::atomic<long long> query_lat_ns_total{0};  // 所有 query 独立计时的总和

    std::vector<std::thread> ths;
    ths.reserve(num_threads);
    size_t per = ranges.size() / num_threads;

    auto t0 = Clock::now();
    for (int t = 0; t < num_threads; ++t) {
        size_t L = t * per;
        size_t R = (t == num_threads - 1) ? ranges.size() : (t + 1) * per;
        ths.emplace_back(worker_range_pgm_disk<Epsilon>,
                         std::cref(index),
                         fd,
                         std::cref(ranges),
                         L, R,
                         std::ref(logical_ios),
                         std::ref(physical_ios),
                         std::ref(io_time_ns),
                         std::ref(query_lat_ns_total));
    }
    for (auto& th : ths) th.join();
    auto t1 = Clock::now();

    long long wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    BenchmarkResult out{};
    out.epsilon        = Epsilon;
    out.wall_ns        = wall_ns;
    out.avg_latency_ns = double(query_lat_ns_total.load()) / std::max<size_t>(1, ranges.size());
    out.hit_ratio      = 0.0;
    out.data_io_ns     = io_time_ns.load();
    out.height         = index.height();
    out.logical_ios    = logical_ios.load();
    out.physical_ios   = physical_ios.load();

    ::close(fd);
    return out;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset_base> [n_keys]\n"
                  << "  e.g. " << argv[0]
                  << " fb_200M_uint64_unique 200000000\n";
        return 1;
    }

    const std::string dataset  = argv[1];               // 如：fb_200M_uint64_unique
    const int N_KEYS = (argc >= 3) ? std::stoi(argv[2])
                                   : 200000000;        // 默认 200M，可按需改

    const std::string datafile  = std::string(DATASETS) + dataset;
    const std::string rangefile = std::string(DATASETS) + dataset + ".range.bin";

    const size_t repeat  = 3;
    const size_t EPSILON = 16;    // 这里示例固定 eps=16

    // 加载数据与 range 查询（range 用 safe 版本过滤掉 SENTINEL）
    auto data   = load_data_pgm_safe<KeyType>(datafile, N_KEYS);
    auto ranges = load_ranges_pgm_safe(rangefile);
    if (data.empty() || ranges.empty()) {
        std::cerr << "load data or ranges failed\n";
        return 1;
    }

    std::ofstream csv(dataset + "_pgm_disk_range.csv",
                      std::ios::out | std::ios::trunc);
    if (!csv) {
        std::cerr << "open csv failed\n";
        return 1;
    }

    csv << "baseline,epsilon,threads,avg_latency_ns,wall_s,hit_ratio,"
           "avg_IOs,physical_IOs,data_io_ns\n";
    csv << std::fixed << std::setprecision(6);

    for (size_t r = 0; r < repeat; ++r) {
        for (int threads : {1,2,4,8,16,32,64,128,256,512,1024}) {
            BenchmarkResult res = bench_range_pgm_disk<EPSILON>(data, ranges, datafile, threads);

            std::cout << "[PGM-disk][T=" << threads << "] eps=" << res.epsilon
                      << " | avg="  << res.avg_latency_ns << " ns"
                      << " | wall=" << (res.wall_ns / 1e9) << " s"
                      << " | logical_IOs="  << res.logical_ios
                      << " | physical_IOs=" << res.physical_ios
                      << " | io_ns="        << res.data_io_ns
                      << std::endl;

            csv << "PGM-disk," << res.epsilon << "," << threads << ","
                << res.avg_latency_ns << "," << (res.wall_ns / 1e9) << ","
                << res.hit_ratio << "," << res.logical_ios << ","
                << res.physical_ios << ","
                << res.data_io_ns << "\n";
            csv.flush();
        }
    }

    csv.close();
    return 0;
}
