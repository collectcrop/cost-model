// aulid_range_multithread.cpp
// Compile: g++ -O3 -std=c++17 aulid_range_multithread.cpp -o aulid_range_multithread -pthread
//
// Usage:
//   ./aulid_range_multithread <dataset_basename> <num_keys> [max_log2_threads] [repeats] [index_name]
//
// Example:
//   ./aulid_range_multithread wiki_ts_200M_uint64_unique 200000000 10 3 ./index_aulid_range

#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <iomanip>

#include "aulid/storage_management.h"
#include "utils/include.hpp"
#include "utils/utils.hpp"   // 里边应有 pgm::RangeQ / load_ranges_pgm_safe

#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"

using namespace std::chrono;
using u64       = uint64_t;
using KeyType   = uint64_t;
using ValueType = uint64_t;

struct RunStats {
    double avg_latency_ns        = 0.0;
    double wall_seconds          = 0.0;
    double avg_block_per_query   = 0.0; // avg bc
    double avg_inblock_per_query = 0.0; // avg ic
    double avg_results_per_query = 0.0; // avg #keys in [lo,hi]
};

// === 你之前的 bulk build 封装，直接复用 ===
static void blipp_bulk(LIPPBTree<KeyType, ValueType> *index,
                       int memory_type, const char *index_name,
                       const char *key_path, int count) {
    index->init(const_cast<char*>(index_name), true, memory_type);

    std::ifstream fin(key_path, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open data file: " << key_path << "\n";
        std::exit(1);
    }

    std::vector<KeyType> raw_keys(count);
    fin.read(reinterpret_cast<char*>(raw_keys.data()),
             sizeof(KeyType) * count);
    fin.close();

    // 去重（已排序）
    std::vector<KeyType> keys;
    keys.reserve(raw_keys.size());
    if (!raw_keys.empty()) {
        keys.push_back(raw_keys[0]);
        for (size_t i = 1; i < raw_keys.size(); ++i) {
            if (raw_keys[i] != raw_keys[i - 1]) {
                keys.push_back(raw_keys[i]);
            }
        }
    }
    int unique_cnt = static_cast<int>(keys.size());
    size_t dup_cnt = raw_keys.size() - keys.size();

    std::cout << "total keys=" << raw_keys.size()
              << ", unique keys=" << keys.size()
              << ", duplicates=" << dup_cnt
              << " (" << 100.0 * dup_cnt / raw_keys.size() << "%)" << std::endl;

    ValueType *values = new ValueType[unique_cnt];
    for (int i = 0; i < unique_cnt; i++) {
        values[i] = static_cast<ValueType>(i);
    }

    std::cout << "start to build... " << std::endl;
    auto bulk_start = high_resolution_clock::now();
    index->bulk_load_entry(keys.data(), values, unique_cnt);
    auto bulk_end   = high_resolution_clock::now();
    long long bulk_ns =
        duration_cast<nanoseconds>(bulk_end - bulk_start).count();
    std::cout << "bulk load time: " << (bulk_ns / 1e9) << " s" << std::endl;

    delete[] values;
}

// === range worker：每个线程处理一部分 RangeQ 列表 ===
static void range_worker(
    LIPPBTree<KeyType, ValueType>* index,
    const std::vector<pgm::RangeQ>& ranges,
    std::atomic<u64>& total_latency_ns,
    std::atomic<u64>& total_blocks,
    std::atomic<u64>& total_inblocks,
    std::atomic<u64>& processed_count,
    std::atomic<u64>& total_results)
{
    for (const auto& rq : ranges) {
        KeyType lo = rq.lo;
        KeyType hi = rq.hi;
        if (lo > hi) std::swap(lo, hi);

        int bc = 0;
        int ic = 0;

        auto t0 = steady_clock::now();
        // ★ 新增的 [lo,hi] 范围接口：统计区间内 keys 个数 & I/O
        int cnt = index->lippb_range_entry(lo, hi, &bc, &ic);
        auto t1 = steady_clock::now();

        u64 lat = (u64)duration_cast<nanoseconds>(t1 - t0).count();

        total_latency_ns.fetch_add(lat,   std::memory_order_relaxed);
        total_blocks.fetch_add((u64)bc,   std::memory_order_relaxed);
        total_inblocks.fetch_add((u64)ic, std::memory_order_relaxed);
        total_results.fetch_add((u64)cnt, std::memory_order_relaxed);
        processed_count.fetch_add(1,      std::memory_order_relaxed);
    }
}

// === 单次实验：固定线程数，对全体 ranges 做一次完整测试 ===
static RunStats run_once_range(
    LIPPBTree<KeyType, ValueType>* index,
    const std::vector<pgm::RangeQ>& all_ranges,
    int thread_count)
{
    RunStats rs;

    int actual_threads = thread_count;
    if ((size_t)actual_threads > all_ranges.size())
        actual_threads = (int)all_ranges.size();
    if (actual_threads <= 0) actual_threads = 1;

    // 均匀切分 ranges
    std::vector<std::vector<pgm::RangeQ>> parts(actual_threads);
    size_t chunk = all_ranges.size() / actual_threads;
    for (int t = 0; t < actual_threads; ++t) {
        size_t begin = t * chunk;
        size_t end   = (t == actual_threads - 1)
                     ? all_ranges.size()
                     : begin + chunk;
        parts[t].assign(all_ranges.begin() + begin,
                        all_ranges.begin() + end);
    }

    std::atomic<u64> total_latency_ns{0};
    std::atomic<u64> total_blocks{0};
    std::atomic<u64> total_inblocks{0};
    std::atomic<u64> processed_count{0};
    std::atomic<u64> total_results{0};

    std::vector<std::thread> ths;
    auto wall0 = steady_clock::now();
    for (int t = 0; t < actual_threads; ++t) {
        ths.emplace_back(range_worker, index, std::cref(parts[t]),
                         std::ref(total_latency_ns),
                         std::ref(total_blocks),
                         std::ref(total_inblocks),
                         std::ref(processed_count),
                         std::ref(total_results));
    }
    for (auto &th : ths) th.join();
    auto wall1 = steady_clock::now();

    double wall_s = duration<double>(wall1 - wall0).count();
    u64 proc      = processed_count.load();
    u64 lat_ns    = total_latency_ns.load();
    u64 blocks    = total_blocks.load();
    u64 inblocks  = total_inblocks.load();
    u64 results   = total_results.load();

    rs.wall_seconds          = wall_s;
    rs.avg_latency_ns        = proc ? (double)lat_ns   / (double)proc : 0.0;
    rs.avg_block_per_query   = proc ? (double)blocks   / (double)proc : 0.0;
    rs.avg_inblock_per_query = proc ? (double)inblocks / (double)proc : 0.0;
    rs.avg_results_per_query = proc ? (double)results  / (double)proc : 0.0;
    return rs;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " <dataset_basename> <num_keys> [max_log2_threads] [repeats] [index_name]\n\n"
                  << "Example:\n  " << argv[0]
                  << " wiki_ts_200M_uint64_unique 200000000 10 3 ./index_aulid_range\n";
        return 1;
    }

    // 必选参数
    std::string dataset_basename = argv[1];
    uint64_t num_keys  = std::strtoull(argv[2], nullptr, 10);
    (void)num_keys; // 当前没用到，如需 sanity check 可用

    // 可选：最大 log2(threads)
    int max_exp = 10;
    if (argc >= 4) {
        max_exp = std::atoi(argv[3]);
        if (max_exp < 0) max_exp = 0;
    }

    // 可选：每个线程数重复次数
    int repeats = 3;
    if (argc >= 5) {
        repeats = std::atoi(argv[4]);
        if (repeats <= 0) repeats = 1;
    }

    // 可选：索引文件名
    std::string index_name_str = "./index_aulid_range";
    if (argc >= 6) {
        index_name_str = argv[5];
    }

    // 路径拼接
    std::string data_file  = std::string(DATASETS) + dataset_basename;
    std::string range_file = std::string(DATASETS) + dataset_basename + ".range.bin";

    char *index_name_c = const_cast<char*>(index_name_str.c_str());
    char *data_file_c  = const_cast<char*>(data_file.c_str());

    // 构建 AULID 磁盘索引
    std::cout << "[AULID-Range] Building index from " << data_file
              << " with " << num_keys << " keys, index file = "
              << index_name_str << "\n";

    LIPPBTree<KeyType, ValueType> index;
    blipp_bulk(&index, LEAF_DISK, index_name_c, data_file_c, (int)num_keys);

    // 加载 RangeQ workload（带 SENTINEL 过滤）
    std::vector<pgm::RangeQ> all_ranges = load_ranges_pgm_safe(range_file);
    if (all_ranges.empty()) {
        std::cerr << "No ranges loaded from " << range_file << "\n";
        return 1;
    }
    std::cout << "Loaded ranges from " << range_file
              << ", count = " << all_ranges.size() << std::endl;

    // 输出 CSV
    std::string csv_name = dataset_basename + "_aulid_range_multithread.csv";
    std::ofstream csv(csv_name, std::ios::out | std::ios::trunc);
    if (!csv) {
        std::cerr << "Failed to open CSV output: " << csv_name << "\n";
        return 1;
    }
    csv << "baseline,threads,latency_ns,walltime_s,avg_IOs,avg_inblock,avg_results\n";
    csv << std::fixed << std::setprecision(6);

    // 遍历线程数 1,2,4,...,2^max_exp
    for (int e = 0; e <= max_exp; ++e) {
        uint64_t threads = 1ULL << e;
        if (threads > all_ranges.size()) threads = all_ranges.size();
        if (threads == 0) threads = 1;

        std::cout << "Testing threads=" << threads
                  << " repeats=" << repeats << " ...\n";

        for (int r = 0; r < repeats; ++r) {
            RunStats rs = run_once_range(&index, all_ranges,
                                         static_cast<int>(threads));

            std::cout << "  run " << (r + 1)
                      << " threads=" << threads
                      << " avg_lat_ns=" << rs.avg_latency_ns
                      << " wall_s=" << rs.wall_seconds
                      << " avg_block=" << rs.avg_block_per_query
                      << " avg_inblock=" << rs.avg_inblock_per_query
                      << " avg_results=" << rs.avg_results_per_query
                      << "\n";

            csv << "AULID," << threads << ","
                << rs.avg_latency_ns << ","
                << rs.wall_seconds  << ","
                << rs.avg_block_per_query   << ","
                << rs.avg_inblock_per_query << ","
                << rs.avg_results_per_query << "\n";
            csv.flush();
        }
    }

    csv.close();
    std::cout << "Finished. Results saved to " << csv_name << "\n";
    return 0;
}
