// parallel_rmi_range.cpp: FALCON + RMI 多线程 range benchmark (multi-dataset version)

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
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <fcntl.h>
#include <future>
#include <iomanip>
#include <string>

#include "distribution/zipf.hpp"
#include "utils/include.hpp"
#include "utils/utils.hpp"
#include "utils/LatencyRecorder.hpp"
#include "FALCON/Falcon.hpp"
#include "cache/CacheInterface.hpp"

// dataset-specific RMI headers
#include "rmi/books_rmi.h"
#include "rmi/fb_rmi.h"
#include "rmi/osm_rmi.h"
#include "rmi/wiki_rmi.h"

using KeyType = uint64_t;

#define DIRECTORY       "/mnt/home/zwshi/learned-index/cost-model/experiments/"
#define DATASETS        "/mnt/home/zwshi/Datasets/SOSD/"
#define RMI_PARAMETERS  "/mnt/home/zwshi/learned-index/cost-model/include/rmi/rmi_data/"

struct BenchmarkResult {
    size_t epsilon;   // RMI 不需要 epsilon，用 0 占位
    double time_ns;   // 平均 query latency (ns)
    double hit_ratio; // cache hit ratio
    time_t total_time;   // 总 wall time (ns)
    time_t data_IO_time; // I/O 时间 (ns)
    time_t index_time;   // index 时间 (ns)
    time_t cache_time;   // cache 时间 (ns)
    size_t height;       // RMI “高度”，这里只有 2 层，填 2
    size_t data_IOs;     // 物理 I/O 次数 (merged)
};

using timer = std::chrono::high_resolution_clock;
std::atomic<size_t> global_queries_done{0};

// ============================================================
// 1) Dataset type detection
// ============================================================

enum class DatasetType {
    BOOKS,
    FB,
    OSM,
    WIKI,
    UNKNOWN
};

static inline bool starts_with(const std::string& s, const std::string& pref) {
    return s.size() >= pref.size() && s.compare(0, pref.size(), pref) == 0;
}

DatasetType detect_dataset_type(const std::string& basename) {
    if (starts_with(basename, "books")) return DatasetType::BOOKS;
    if (starts_with(basename, "fb"))    return DatasetType::FB;
    if (starts_with(basename, "osm"))   return DatasetType::OSM;
    if (starts_with(basename, "wiki"))  return DatasetType::WIKI;
    return DatasetType::UNKNOWN;
}

const char* dataset_type_name(DatasetType t) {
    switch (t) {
        case DatasetType::BOOKS: return "books";
        case DatasetType::FB:    return "fb";
        case DatasetType::OSM:   return "osm";
        case DatasetType::WIKI:  return "wiki";
        default:                 return "unknown";
    }
}

// ============================================================
// 2) Traits for different RMI models
// ============================================================

struct BooksTraits {
    static bool load(const std::string& root) { return books_rmi::load(root.c_str()); }
    static void cleanup() { books_rmi::cleanup(); }
    static inline size_t lookup(uint64_t key, size_t* err) {
        return (size_t)books_rmi::lookup(key, err);
    }
    static constexpr const char* name = "books_rmi";
};

struct FbTraits {
    static bool load(const std::string& root) { return fb_rmi::load(root.c_str()); }
    static void cleanup() { fb_rmi::cleanup(); }
    static inline size_t lookup(uint64_t key, size_t* err) {
        return (size_t)fb_rmi::lookup(key, err);
    }
    static constexpr const char* name = "fb_rmi";
};

struct OsmTraits {
    static bool load(const std::string& root) { return osm_rmi::load(root.c_str()); }
    static void cleanup() { osm_rmi::cleanup(); }
    static inline size_t lookup(uint64_t key, size_t* err) {
        return (size_t)osm_rmi::lookup(key, err);
    }
    static constexpr const char* name = "osm_rmi";
};

struct WikiTraits {
    static bool load(const std::string& root) { return wiki_rmi::load(root.c_str()); }
    static void cleanup() { wiki_rmi::cleanup(); }
    static inline size_t lookup(uint64_t key, size_t* err) {
        return (size_t)wiki_rmi::lookup(key, err);
    }
    static constexpr const char* name = "wiki_rmi";
};

// ============================================================
// 3) Generic RMI adapter for FalconRMI
// ============================================================

template <typename Traits>
struct GenericRMIIndex {
    using key_type = uint64_t;

    explicit GenericRMIIndex(size_t n) : n_(n) {}

    inline size_t lookup(uint64_t key, size_t *err) const {
        return Traits::lookup(key, err);
    }

    inline size_t size() const { return n_; }

    inline size_t height() const { return 2; } // 2-level RMI

private:
    size_t n_;
};

// ============================================================
// 4) Worker thread for range queries
// ============================================================
//
// 要求：falcon::FalconRMI 暴露接口：
//   std::future<falcon::RangeResult> range_lookup(uint64_t lo, uint64_t hi);
//
// RangeResult 里包含所有命中的 keys（和 FalconPGM 的实现一致）。
//
template <typename FalconT>
void worker_thread_range(FalconT* F,
                         const std::vector<pgm::RangeQ>& ranges,
                         size_t begin, size_t end,
                         std::atomic<long long>& total_time_ns,
                         LatencyRecorder* latency) {
    using clock_t = std::chrono::high_resolution_clock;
    constexpr size_t BATCH = 128;

    struct Pending {
        size_t qid;
        clock_t::time_point t0;
        std::future<falcon::RangeResult> fut;
    };

    auto thread_t0 = clock_t::now();

    std::vector<Pending> inflight;
    inflight.reserve(BATCH);

    size_t i = begin;
    while (i < end) {
        inflight.clear();
        const size_t j = std::min(end, i + BATCH);

        // 提交一批 range_lookup
        for (; i < j; ++i) {
            const auto &q = ranges[i];
            auto t0 = clock_t::now();
            auto fut = F->range_lookup(q.lo, q.hi);
            inflight.push_back(Pending{ i, t0, std::move(fut) });
        }

        // 收割 futures
        for (auto &p : inflight) {
            auto res = p.fut.get();
            (void)res; // 这里只是触发完整管线，不额外处理结果
            global_queries_done.fetch_add(1, std::memory_order_relaxed);
        }
    }

    auto thread_t1 = clock_t::now();
    auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(thread_t1 - thread_t0).count();
    latency->set(begin, dt);
    total_time_ns.fetch_add(dt, std::memory_order_relaxed);
}

// ============================================================
// 5) Generic benchmark for any dataset RMI (range queries)
// ============================================================

template <typename Traits>
BenchmarkResult benchmark_mt_rmi_range_generic(std::vector<KeyType> data,
                                               std::vector<pgm::RangeQ> ranges,
                                               const std::string &filename,
                                               pgm::CachePolicy s,
                                               int num_threads,
                                               size_t memory_budget_bytes) {
    // RMI 参数只 load 一次
    static std::once_flag load_flag;
    std::call_once(load_flag, [&](){
        if (!Traits::load(RMI_PARAMETERS)) {
            std::cerr << Traits::name << "::load failed\n";
            std::exit(1);
        }
    });

    GenericRMIIndex<Traits> index(data.size());

    int data_fd = ::open(filename.c_str(), O_RDONLY | O_DIRECT);
    if (data_fd < 0) {
        perror("open data");
        std::exit(1);
    }

    pgm::CachePolicy policy = pgm::CachePolicy::NONE;
    switch (s) {
        case pgm::CachePolicy::LRU:  policy = pgm::CachePolicy::LRU;  break;
        case pgm::CachePolicy::FIFO: policy = pgm::CachePolicy::FIFO; break;
        case pgm::CachePolicy::LFU:  policy = pgm::CachePolicy::LFU;  break;
        case pgm::CachePolicy::NONE: policy = pgm::CachePolicy::NONE; break;
    }

    // 这里假定你已经有 falcon::FalconRMI 模板类，其构造接口和 point 版本一致
    falcon::FalconRMI<uint64_t, GenericRMIIndex<Traits>> F(
        index,
        data_fd,
        pgm::IO_URING,
        memory_budget_bytes,
        policy,
        /*cache_shards=*/1,
        /*max_pages_per_batch=*/256,
        /*max_wait_us=*/50,
        /*workers=*/ std::min(std::max(num_threads / 8, 1), 16),
        data.size()
    );

    std::atomic<long long> total_time_ns{0};
    global_queries_done = 0;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    size_t per_thread = ranges.size() / num_threads;

    const size_t Q = ranges.size();
    auto *latency = new LatencyRecorder(Q);

    auto start_all = timer::now();
    for (int t = 0; t < num_threads; t++) {
        size_t begin = t * per_thread;
        size_t end   = (t == num_threads - 1) ? ranges.size() : (t + 1) * per_thread;
        threads.emplace_back(
            worker_thread_range<falcon::FalconRMI<uint64_t, GenericRMIIndex<Traits>>>,
            &F, std::ref(ranges),
            begin, end, std::ref(total_time_ns), latency
        );
    }
    for (auto &th : threads) th.join();
    auto end_all = timer::now();

    auto wall_clock_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_all - start_all).count();

    auto st = F.stats();
    double hit_ratio = 0.0;
    auto hm = st.cache_hits + st.cache_misses;
    if (hm) hit_ratio = double(st.cache_hits) / double(hm);

    BenchmarkResult result;
    result.epsilon      = 0;
    result.time_ns      = latency->get_avg();  // 按你的 LatencyRecorder 语义计算平均
    result.hit_ratio    = hit_ratio;
    result.total_time   = wall_clock_ns;
    result.data_IO_time = st.io_ns;
    result.index_time   = st.index_ns;
    result.cache_time   = st.cache_ns;
    result.height       = index.height();
    result.data_IOs     = st.physical_ios;

    ::close(data_fd);
    delete latency;

    // 如果想每次 benchmark 后释放模型，可以在进程结束前调用 Traits::cleanup()
    return result;
}

// ============================================================
// 6) main: parse args & dispatch by dataset type
// ============================================================

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " <dataset_basename> <num_keys> [max_log2_threads] [repeats] [memory_mb] [baseline_name]\n\n"
                  << "Example:\n  " << argv[0]
                  << " books_200M_uint64_unique 200000000 10 3 40 FALCON-RMI-range\n";
        return 1;
    }

    std::string dataset_basename = argv[1];
    uint64_t num_keys = std::strtoull(argv[2], nullptr, 10);

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

    int mem_mb = 40;
    if (argc >= 6) {
        mem_mb = std::atoi(argv[5]);
        if (mem_mb < 0) mem_mb = 0;
    }
    size_t memory_budget_bytes = static_cast<size_t>(mem_mb) * 1024ull * 1024ull;

    std::string baseline = "FALCON-RMI";
    if (argc >= 7) baseline = argv[6];

    DatasetType dtype = detect_dataset_type(dataset_basename);
    if (dtype == DatasetType::UNKNOWN) {
        std::cerr << "Unknown dataset type from basename: " << dataset_basename << "\n"
                  << "Expected prefix: books / fb / osm / wiki\n";
        return 1;
    }
    std::cout << "Detected dataset type: " << dataset_type_name(dtype) << "\n";

    std::string filename     = dataset_basename;
    std::string range_fname  = dataset_basename + ".range.bin";
    std::string file         = std::string(DATASETS) + filename;
    std::string range_file   = std::string(DATASETS) + range_fname;

    // data 用于给 GenericRMIIndex 传递 N
    std::vector<KeyType> data   = load_data_pgm_safe<KeyType>(file, num_keys);
    // ranges 使用带 SENTINEL 过滤的 safe 版本
    std::vector<pgm::RangeQ> ranges = load_ranges_pgm_safe(range_file);

    if (data.empty() || ranges.empty()) {
        std::cerr << "data or ranges empty, abort.\n";
        return 1;
    }

    std::string csv_name = dataset_basename + "_falcon_rmi_range.csv";
    std::ofstream csv(csv_name, std::ios::out | std::ios::trunc);
    if (!csv) {
        std::cerr << "Failed to open CSV output: " << csv_name << "\n";
        return 1;
    }

    // 和 point 版本保持相同的列格式，便于对比
    csv << "baseline,threads,latency_ns,wall_time_s,avg_IOs,data_IO_time\n";
    csv << std::fixed << std::setprecision(6);

    for (int r = 0; r < repeats; ++r) {
        for (int e = 0; e <= max_exp; ++e) {
            uint64_t threads = 1ULL << e;
            if (threads == 0) threads = 1;

            BenchmarkResult result;

            switch (dtype) {
                case DatasetType::BOOKS:
                    result = benchmark_mt_rmi_range_generic<BooksTraits>(
                        data, ranges, file,
                        pgm::CachePolicy::NONE,
                        (int)threads, memory_budget_bytes
                    );
                    break;
                case DatasetType::FB:
                    result = benchmark_mt_rmi_range_generic<FbTraits>(
                        data, ranges, file,
                        pgm::CachePolicy::NONE,
                        (int)threads, memory_budget_bytes
                    );
                    break;
                case DatasetType::OSM:
                    result = benchmark_mt_rmi_range_generic<OsmTraits>(
                        data, ranges, file,
                        pgm::CachePolicy::NONE,
                        (int)threads, memory_budget_bytes
                    );
                    break;
                case DatasetType::WIKI:
                    result = benchmark_mt_rmi_range_generic<WikiTraits>(
                        data, ranges, file,
                        pgm::CachePolicy::NONE,
                        (int)threads, memory_budget_bytes
                    );
                    break;
                default:
                    std::cerr << "Unexpected dataset type.\n";
                    return 1;
            }

            std::cout << "[Threads=" << threads << "] "
                      << "avg query time=" << result.time_ns << " ns"
                      << ", hit ratio=" << result.hit_ratio
                      << ", total wall time=" << result.total_time / 1e9 << " s"
                      << ", data IOs=" << result.data_IOs
                      << ", data IO time=" << result.data_IO_time / 1e9 << " s"
                      << std::endl;

            csv << baseline << "," << threads << ","
                << result.time_ns << ","
                << (result.total_time / 1e9) << ","   // wall_time_s
                << result.data_IOs << ","
                << result.data_IO_time << "\n";
            csv.flush();
        }
    }

    csv.close();
    std::cout << "Finished. Results saved to " << csv_name << "\n";
    return 0;
}
