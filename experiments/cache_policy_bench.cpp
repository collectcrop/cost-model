// pgm_cache_bench.cpp
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <unordered_set>
#include "pgm/pgm_index.hpp"
#include "utils/include.hpp"
#include "utils/utils.hpp"

#include "IO/io_interface.hpp"
#include "IO/SyncInterface.hpp"
// 你也可以改成 #include "IO/IOuringInterface.hpp"

#include "cache/ShardedFIFOCache.hpp"
#include "cache/ShardedLRUCache.hpp"
#include "cache/ShardedLFUCache.hpp"

using KeyType = uint64_t;
using timer   = std::chrono::high_resolution_clock;
using ns      = std::chrono::nanoseconds;

// 根据你的习惯改
#define DATASETS  "/mnt/home/zwshi/Datasets/SOSD/"
#define DATA_FILE "osm_cellids_200M_uint64_unique"   // 纯 key 的二进制文件
// #define QUERY_FILE "osm_cellids_200M_uint64_unique.query.bin"
#define OUTPUT_FILE "cache_bench_results.csv"
#define QUERY_FILE "osm_cellids_200M_uint64_unique.1Mtable.bin"
#define TOTAL_KEYS 200000000ull
bool first_run = true;
// ================ 工具：从 key 映射到 page_idx ================

// 以 PGMIndex 的 ApproxPos 搜索结果为例：pos 是近似下标
template <size_t Epsilon>
inline size_t key_to_page_idx(const pgm::PGMIndex<KeyType, Epsilon> &index,
                              KeyType key,
                              size_t n_keys) {
    auto approx = index.search(key);
    size_t pos = approx.pos;
    if (pos >= n_keys) pos = n_keys - 1;
    return pos / pgm::ITEM_PER_PAGE;
}

template <size_t Epsilon>
size_t count_unique_pages_in_queries(
    const pgm::PGMIndex<KeyType, Epsilon> &pgm_index,
    const std::vector<KeyType> &queries,
    size_t n_keys)
{
    const size_t max_page_idx = (n_keys > 0) ? (n_keys - 1) / pgm::ITEM_PER_PAGE : 0;
    std::unordered_set<size_t> S;
    S.reserve(1 << 20);

    for (auto key : queries) {
        auto [lo, hi] = pgm_index.estimate_pages_for_key(key);
        if (hi > max_page_idx) hi = max_page_idx;
        for (size_t p = lo; p <= hi; ++p) {
            S.insert(p);
        }
    }
    return S.size();
}
std::vector<KeyType> gen_hotspot_queries(const std::vector<KeyType> &keys,
                                         size_t num_queries,
                                         double hot_ratio = 0.2,
                                         double hot_prob  = 0.8,
                                         uint64_t seed    = 42) {
    std::mt19937_64 rng(seed);
    size_t n = keys.size();
    size_t hot_size = std::max<size_t>(1, n * hot_ratio);

    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_int_distribution<size_t> hot_dist(0, hot_size - 1);
    std::uniform_int_distribution<size_t> cold_dist(hot_size, n - 1);

    std::vector<KeyType> q;
    q.reserve(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        size_t idx;
        if (prob(rng) < hot_prob) idx = hot_dist(rng);
        else idx = cold_dist(rng);
        q.push_back(keys[idx]);
    }
    return q;
}

// 顺序查询：模拟 ordered scan
std::vector<KeyType> gen_sequential_queries(const std::vector<KeyType> &keys,
                                            size_t num_queries) {
    std::vector<KeyType> q;
    q.reserve(num_queries);
    size_t n = keys.size();
    for (size_t i = 0; i < num_queries; ++i) {
        q.push_back(keys[i % n]);
    }
    return q;
}


// ================ 结果结构体 =================

struct BenchResult {
    std::string policy;
    std::string workload;
    size_t cache_MiB;
    size_t cache_pages;
    size_t ops;
    double hit_ratio;
    double avg_latency_ns;      // 全部 query 的平均延迟 (cache+IO)
    double avg_miss_io_ns;      // 只对 miss 的平均 IO 延迟
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t io_bytes;
    uint64_t io_physical_ios;
};

// ================ 通用 benchmark 函数 =================

// CacheType 需要提供：
//   bool get(size_t page_idx, pgm::Page& out);
//   void put(size_t page_idx, pgm::Page&& page);
//   CacheStats& stats();
template <typename CacheType, size_t Epsilon>
BenchResult run_pgm_cache_bench(
    const std::string& policy_name,
    const std::string& workload_name,
    CacheType& cache,
    IOInterface& io,
    const pgm::PGMIndex<KeyType, Epsilon>& pgm_index,
    const std::vector<KeyType>& queries,
    size_t n_keys,
    double warmup_ratio = 0.0)
{
    using namespace std;
    using namespace std::chrono;

    const size_t Q = queries.size();
    size_t warmup_ops = static_cast<size_t>(Q * warmup_ratio);
    if (warmup_ops > Q) warmup_ops = Q;

    const size_t max_page_idx =
        (n_keys > 0) ? (n_keys - 1) / pgm::ITEM_PER_PAGE : 0;

    // ---------- 1) warm-up phase：仍然不统计延迟 ----------
    for (size_t i = 0; i < warmup_ops; ++i) {
        KeyType key = queries[i];

        // 关键修改：PGM 给出 [pos±ε] 的 page 范围
        auto [page_lo, page_hi] = pgm_index.estimate_pages_for_key(key);
        if (page_hi > max_page_idx) page_hi = max_page_idx;

        for (size_t page_idx = page_lo; page_idx <= page_hi; ++page_idx) {
            pgm::Page out;
            if (!cache.get(page_idx, out)) {
                // miss: 触发真实 IO
                auto [p, r] = io.triggerIO(page_idx);
                cache.put(page_idx, std::move(p));
            }
        }
    }

    // ---------- 2) measurement phase ----------
    uint64_t io_ns           = 0;
    uint64_t io_bytes        = 0;
    uint64_t io_physical_ios = 0;

    auto t0 = timer::now();
    for (size_t i = warmup_ops; i < Q; ++i) {
        KeyType key = queries[i];

        auto [page_lo, page_hi] = pgm_index.estimate_pages_for_key(key);
        if (page_hi > max_page_idx) page_hi = max_page_idx;

        // 每个 query 对 [page_lo,page_hi] 范围内所有 page 做 cache + IO
        for (size_t page_idx = page_lo; page_idx <= page_hi; ++page_idx) {
            pgm::Page out;
            if (!cache.get(page_idx, out)) {
                auto [p, r] = io.triggerIO(page_idx);
                io_ns           += r.ns;
                io_bytes        += r.bytes;
                io_physical_ios += r.physical_ios;
                cache.put(page_idx, std::move(p));
            }
        }
    }
    auto t1 = timer::now();

    // 以“query 数量”为单位统计平均延迟
    uint64_t measured_ops = Q - warmup_ops;
    double total_ns       = duration_cast<ns>(t1 - t0).count();
    double avg_latency_ns = measured_ops ? total_ns / double(measured_ops) : 0.0;

    // 从 cache 中读命中统计（page 级 hits/misses）
    const auto& st = cache.stats();
    uint64_t hits   = st.hits.load(std::memory_order_relaxed);
    uint64_t misses = st.misses.load(std::memory_order_relaxed);
    uint64_t evicts = st.evictions.load(std::memory_order_relaxed);

    double hit_ratio = (hits + misses) ? double(hits) / double(hits + misses) : 0.0;
    double avg_miss_io_ns = misses ? double(io_ns) / double(misses) : 0.0;

    BenchResult br{};
    br.policy           = policy_name;
    br.workload         = workload_name;
    br.cache_MiB        = 0;           // 由外层 main 填
    br.cache_pages      = 0;           // 由外层 main 填
    br.ops              = measured_ops;
    br.hit_ratio        = hit_ratio;
    br.avg_latency_ns   = avg_latency_ns;   // 每个 query 的平均端到端时间
    br.avg_miss_io_ns   = avg_miss_io_ns;   // 每个 page miss 的平均 IO 时间
    br.hits             = hits;
    br.misses           = misses;
    br.evictions        = evicts;
    br.io_bytes         = io_bytes;
    br.io_physical_ios  = io_physical_ios;
    return br;
}

// ================ main：扫缓存大小+三种策略 =================

int main() {
    using namespace std;

    // 1) 加载数据（纯 key 文件）
    string data_file = string(DATASETS) + DATA_FILE;
    string query_file = string(DATASETS) + QUERY_FILE;
    cout << "Loading data from " << data_file << endl;
    vector<KeyType> keys = load_data(data_file, TOTAL_KEYS);
    if (keys.empty()) {
        cerr << "load_data failed\n";
        return 1;
    }
    size_t n_keys = keys.size();

    // 2) 构建 PGMIndex
    constexpr size_t Epsilon = 32;
    cout << "Building PGMIndex with epsilon = " << Epsilon << endl;
    pgm::PGMIndex<KeyType, Epsilon> pgm_index(keys);

    // 3) 打开数据文件（按 page_idx 映射）
    int fd = ::open(data_file.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open data file");
        return 1;
    }

    // 4) 选择 IO 后端
    SyncInterface io(fd);
    // 或者：IoUringInterface io(fd, /*queue_depth=*/256);

    // 5) 生成查询
    // size_t NUM_QUERIES = 1'000'000;
    // auto hotspot_q   = gen_hotspot_queries(keys, NUM_QUERIES, 0.2, 0.8, 42);
    // auto seq_q       = gen_sequential_queries(keys, NUM_QUERIES);
    auto q           = load_queries(query_file);
    size_t uniq_pages = count_unique_pages_in_queries<Epsilon>(pgm_index, q, n_keys);
    std::cout << "Unique pages touched by queries = " << uniq_pages << std::endl;

    // 6) 缓存容量 sweep（MiB）
    vector<double> cache_sizes_b = {1e5, 1e6, 32e6, 512e6, 4096e6}; // 1MiB, 32MiB, 512MiB, 4GiB

    ofstream result_file;
    if (first_run) {
        result_file.open(OUTPUT_FILE, ios::out);
        if (result_file.is_open()) {
            result_file << "policy,workload,cache_MiB,cache_pages,ops,"
                        << "hit_ratio,avg_latency_ns,avg_miss_io_ns,"
                        << "hits,misses,evictions,io_bytes,io_physical_ios\n";
            result_file.close();
            first_run = false;
        }
    }
    const int NUM_RUNS = 5;
    
    for (int run_id = 0; run_id < NUM_RUNS; ++run_id) { 
        cout << std::fixed << std::setprecision(4);
        cout << "policy,workload,cache_MiB,cache_pages,ops,"
                "hit_ratio,avg_latency_ns,avg_miss_io_ns,"
                "hits,misses,evictions,io_bytes,io_physical_ios\n";

        for (auto cache_b : cache_sizes_b){
            size_t cache_pages = (size_t)cache_b / pgm::PAGE_SIZE;
            if (cache_pages == 0) cache_pages = 1;
            size_t shards = 1;  

            // ---------- simple workload ----------
            {
                pgm::ShardedFIFOCache fifo(cache_pages, shards);
                pgm::ShardedLRUCache  lru (cache_pages, shards);
                pgm::ShardedLFUCache  lfu (cache_pages, shards);

                auto r_fifo = run_pgm_cache_bench<pgm::ShardedFIFOCache, Epsilon>(
                        "FIFO", "simple", fifo, io, pgm_index, q, n_keys);
                auto r_lru  = run_pgm_cache_bench<pgm::ShardedLRUCache,  Epsilon>(
                        "LRU",  "simple", lru,  io, pgm_index, q, n_keys);
                auto r_lfu  = run_pgm_cache_bench<pgm::ShardedLFUCache,  Epsilon>(
                        "LFU",  "simple", lfu,  io, pgm_index, q, n_keys);

                r_fifo.cache_MiB   = cache_b;
                r_fifo.cache_pages = cache_pages;
                r_lru.cache_MiB    = cache_b;
                r_lru.cache_pages  = cache_pages;
                r_lfu.cache_MiB    = cache_b;
                r_lfu.cache_pages  = cache_pages;

                result_file.open(OUTPUT_FILE, ios::app);
                for (auto &r : {r_fifo, r_lru, r_lfu}) {
                    stringstream ss;
                    ss << r.policy << "," << r.workload << ","
                       << r.cache_MiB/1e6 << "," << r.cache_pages << ","
                       << r.ops << ","
                       << r.hit_ratio << "," << r.avg_latency_ns << ","
                       << r.avg_miss_io_ns << ","
                       << r.hits << "," << r.misses << "," << r.evictions << ","
                       << r.io_bytes << "," << r.io_physical_ios;
                    
                    // 控制台输出
                    cout << ss.str() << "\n";
                    
                    // 文件输出
                    if (result_file.is_open()) {
                        result_file << ss.str() << "\n";
                    }
                }
                if (result_file.is_open()) {
                    result_file.close();
                }
            }
        }
    }

    ::close(fd);
    return 0;
}
