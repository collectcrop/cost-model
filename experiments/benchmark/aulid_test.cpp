// aulid_multithread.cpp
// Compile: g++ -O3 -std=c++17 aulid_multithread.cpp -o aulid_multithread -pthread
// Usage: ./aulid_multithread <index_name> <query_file> <has_size(0|1)> <max_exp> <repeats>

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

// include LIPPBTree header(s)
#include "storage_management.h" 
#include "utils/utils.hpp"
// or the actual header where LIPPBTree is defined
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"

using namespace std::chrono;
using u64 = uint64_t;
using KeyType = uint64_t;    // adapt if your KeyType differs
using ValueType = uint64_t;  // adapt if needed

// wrapper to call your index - adjust template params/namespace as needed
// LIPPBTree<KeyType, ValueType> is used as in your snippet.

struct RunStats {
    double avg_latency_ns = 0;
    double wall_seconds = 0;
    double avg_block_per_lookup = 0; // average bc
    double avg_inblock_per_lookup = 0; // average ic
};

static u64 read_uint64_from_file(std::ifstream &ifs) {
    u64 x;
    ifs.read(reinterpret_cast<char*>(&x), sizeof(u64));
    return x;
}

void blipp_bulk(LIPPBTree<KeyType, ValueType> *index,
                int memory_type, const char *index_name,
                const char *key_path, int count) {
    index->init(const_cast<char*>(index_name), true, memory_type);
    std::ifstream fin(key_path, std::ios::binary);
    // 先读原始数据
    std::vector<KeyType> raw_keys(count);
    fin.read(reinterpret_cast<char*>(raw_keys.data()),
             sizeof(KeyType) * count);
    fin.close();

    // 去重（数据已经是排序好的）
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
    auto bulk_start = std::chrono::high_resolution_clock::now();
    index->bulk_load_entry(keys.data(), values, unique_cnt);
    auto bulk_end = std::chrono::high_resolution_clock::now();
    long long bulk_lookup_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_end - bulk_start).count();
    std::cout << "bulk load time: " << bulk_lookup_time / 1e9 << std::endl;

    delete[] values;
}

// Worker: each线程处理自己的 queries 列表，并累计统计
static void worker_proc(
    LIPPBTree<KeyType, ValueType>* index,
    const std::vector<KeyType>& queries,
    std::atomic<u64>& total_latency_ns,
    std::atomic<u64>& total_blocks,   // sum of bc
    std::atomic<u64>& total_inblocks, // sum of ic
    std::atomic<u64>& processed_count)
{
    ValueType v;
    int not_found_count = 0;
    for (size_t i = 0; i < queries.size(); ++i) {
        KeyType k = queries[i];
        auto t0 = steady_clock::now();
        int bc = 0;
        int ic = 0;
        bool found = index->lippb_search_entry(k, &v, &bc, &ic);
        // optional: you can track misses if needed
        if (!found) {
            not_found_count++;
            // handle not found if necessary
            // std::cout << "not found: " << k << std::endl;
        }
        auto t1 = steady_clock::now();
        u64 lat = (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        total_latency_ns.fetch_add(lat, std::memory_order_relaxed);
        total_blocks.fetch_add((u64)bc, std::memory_order_relaxed);
        total_inblocks.fetch_add((u64)ic, std::memory_order_relaxed);
        processed_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (not_found_count > 0) {
        std::cout << "Total not found in this thread: " << not_found_count << "/" << queries.size() << std::endl;
    }
}

// run single trial with given thread count
static RunStats run_once(
    LIPPBTree<KeyType, ValueType>* index,
    const std::vector<KeyType>& all_queries,
    int thread_count)
{
    RunStats rs;
    // cap thread_count to queries count
    int actual_threads = thread_count;
    if ((size_t)actual_threads > all_queries.size()) actual_threads = (int)all_queries.size();
    if (actual_threads <= 0) actual_threads = 1;

    // split
    std::vector<std::vector<KeyType>> parts(actual_threads);
    size_t chunk = all_queries.size() / actual_threads;
    for (int t = 0; t < actual_threads; ++t) {
        size_t begin = t * chunk;
        size_t end = (t == actual_threads - 1) ? all_queries.size() : begin + chunk;
        parts[t].assign(all_queries.begin() + begin, all_queries.begin() + end);
    }

    std::atomic<u64> total_latency_ns{0};
    std::atomic<u64> total_blocks{0};
    std::atomic<u64> total_inblocks{0};
    std::atomic<u64> processed_count{0};

    // launch threads
    std::vector<std::thread> ths;
    auto wall0 = steady_clock::now();
    for (int t = 0; t < actual_threads; ++t) {
        ths.emplace_back(worker_proc, index, std::cref(parts[t]),
                         std::ref(total_latency_ns), std::ref(total_blocks),
                         std::ref(total_inblocks), std::ref(processed_count));
    }
    for (auto &th : ths) th.join();
    auto wall1 = steady_clock::now();

    double wall_s = std::chrono::duration<double>(wall1 - wall0).count();
    u64 proc = processed_count.load();
    u64 lat_ns = total_latency_ns.load();
    u64 blocks = total_blocks.load();
    u64 inblocks = total_inblocks.load();

    rs.wall_seconds = wall_s;
    rs.avg_latency_ns = proc ? (double)lat_ns / (double)proc : 0.0;
    rs.avg_block_per_lookup = proc ? (double)blocks / (double)proc : 0.0;
    rs.avg_inblock_per_lookup = proc ? (double)inblocks / (double)proc : 0.0;
    return rs;
}

int main(int argc, char** argv) {
    // Usage 提示
    if (argc < 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " <dataset_basename> <num_keys> [max_log2_threads] [repeats] [index_name]\n\n"
                  << "Example:\n  " << argv[0]
                  << " wiki_ts_10M_uint64_unique 10000000 10 3 ./index\n";
        return 1;
    }

    // 1) 必选参数
    std::string dataset_basename = argv[1]; // e.g. wiki_ts_10M_uint64_unique
    uint64_t num_keys = std::strtoull(argv[2], nullptr, 10);

    // 2) 可选：最大 log2(threads)
    int max_exp = 10; // 默认 1..2^14
    if (argc >= 4) {
        max_exp = std::atoi(argv[3]);
        if (max_exp < 0) max_exp = 0;
    }

    // 3) 可选：每个线程数重复次数
    int repeats = 3;
    if (argc >= 5) {
        repeats = std::atoi(argv[4]);
        if (repeats <= 0) repeats = 1;
    }

    // 4) 可选：索引文件名
    std::string index_name_str = "./index";
    if (argc >= 6) {
        index_name_str = argv[5];
    }

    // 5) 拼接数据/查询文件路径
    std::string data_file  = std::string(DATASETS) + dataset_basename;
    std::string query_file = std::string(DATASETS) + dataset_basename + ".query.bin";

    char *index_name_c = const_cast<char*>(index_name_str.c_str());
    char *data_file_c  = const_cast<char*>(data_file.c_str());

    // 6) 构建磁盘索引（bulk_load）
    std::cout << "Building AULID index from " << data_file
              << " with " << num_keys << " keys, index file = " << index_name_str << "\n";
    
    LIPPBTree<KeyType, ValueType> index;
    blipp_bulk(&index, LEAF_DISK, index_name_c, data_file_c, (int)num_keys);

    
    // index.init(index_name_c, false, ALL_DISK);

    // 7) 读取 queries
    std::vector<KeyType> all_queries = load_queries(query_file);
    std::cout << "Loaded queries from " << query_file
              << ", count = " << all_queries.size() << std::endl;

    // 8) 打开 CSV：<dataset_basename>_aulid_multithread.csv
    std::string csv_name = dataset_basename + "_aulid_multithread.csv";
    std::ofstream csv(csv_name, std::ios::out | std::ios::trunc);
    if (!csv) {
        std::cerr << "Failed to open CSV output: " << csv_name << "\n";
        return 1;
    }
    csv << "baseline,threads,latency_ns,walltime_s,avg_IOs\n";
    csv << std::fixed << std::setprecision(6);

    // 9) 遍历线程数（1, 2, 4, ..., 2^max_exp）
    for (int e = 0; e <= max_exp; ++e) {
        uint64_t threads = 1ULL << e;
        if (threads > all_queries.size()) threads = all_queries.size();
        if (threads == 0) threads = 1;

        std::cout << "Testing threads=" << threads
                  << " repeats=" << repeats << " ...\n";


        for (int r = 0; r < repeats; ++r) {
            RunStats rs = run_once(&index, all_queries, static_cast<int>(threads));
            std::cout << "  run " << (r + 1)
                      << " threads=" << threads
                      << " avg_lat_ns=" << rs.avg_latency_ns
                      << " wall_s=" << rs.wall_seconds
                      << " avg_block=" << rs.avg_block_per_lookup
                      << " avg_ic=" << rs.avg_inblock_per_lookup << "\n";

            csv << "AULID," << threads << "," << rs.avg_latency_ns << "," << rs.wall_seconds
                << "," << rs.avg_block_per_lookup << "\n";
            csv.flush();
        }
    }

    csv.close();
    std::cout << "Finished. Results saved to " << csv_name << "\n";
    return 0;
}