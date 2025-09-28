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

void blipp_bulk(int memory_type, char *index_name, char *key_path, int count) {
    LIPPBTree<KeyType, ValueType> index;
    index.init(index_name, true, memory_type);
    std::ifstream fin(key_path, std::ios::binary);
    KeyType *keys = new KeyType[count];
    fin.read((char *) (keys), sizeof(KeyType) * count);
    fin.close();

    ValueType *values = new ValueType[count];
    for (int i = 0; i < count; i++) {
        values[i] = static_cast<ValueType>(i); 
    }
    std::cout << "start to build... " << std::endl;
    std::chrono::high_resolution_clock::time_point bulk_start = std::chrono::high_resolution_clock::now();
    index.bulk_load_entry(keys, values, count);
    std::chrono::high_resolution_clock::time_point bulk_end = std::chrono::high_resolution_clock::now();
    long long bulk_lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(bulk_end - bulk_start).count();
    std::cout << "bulk load time: " << bulk_lookup_time / 1e9 << std::endl;
    // std::cout << "file size:" << index.report_file_size() << " bytes" << std::endl;
    delete[]keys;
    delete[]values;
    return;
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
    std::string query_filename = "books_10M_uint64_unique.query.bin";
    char* file = "/mnt/home/zwshi/Datasets/SOSD/books_10M_uint64_unique";
    std::string query_file = DATASETS + query_filename;
    // if (argc < 6) {
    //     std::cerr << "Usage: " << argv[0] << " <index_name> <query_file> <has_size(0|1)> <max_exp> <repeats>\n";
    //     std::cerr << "Example: " << argv[0] << " myindex /path/queries.bin 1 14 5\n";
    //     return 1;
    // }
    // const char* index_name = argv[1];
    // const char* query_file = argv[2];
    // int has_size = std::atoi(argv[3]);
    // int max_exp = std::atoi(argv[4]);   // e.g. 14
    // int repeats = std::atoi(argv[5]);   // e.g. 5
    char* index_name = "./index";
    int max_exp = 14;  
    int repeats = 5;   
    // 0. construct index
    blipp_bulk(ALL_DISK, index_name, file, 10000000);
    // 1. init index (shared)
    LIPPBTree<KeyType, ValueType> index;
    index.init(const_cast<char*>(index_name), false, ALL_DISK);
    // If your init signature differs, adjust accordingly.

    // 2. read queries from file
    std::vector<KeyType> all_queries = load_queries(query_file);
    std::cout << "Loaded queries: " << all_queries.size() << std::endl;

    // prepare csv
    std::ofstream csv("aulid_multithread.csv", std::ios::out | std::ios::trunc);
    if (!csv) {
        std::cerr << "Failed to open CSV output\n";
        return 1;
    }
    csv << "threads,avg_latency_ns,avg_walltime_s,height,avg_IOs\n";
    csv << std::fixed << std::setprecision(6);

    // 3. loop over power-of-two thread counts
    for (int e = 0; e <= max_exp; ++e) {
        uint64_t threads = 1ULL << e;
        if (threads > all_queries.size()) threads = all_queries.size();
        std::cout << "Testing threads=" << threads << " repeats=" << repeats << " ...\n";

        double sum_lat_ns = 0;
        double sum_wall = 0;
        double sum_height = 0;
        double sum_ios = 0;

        for (int r = 0; r < repeats; ++r) {
            RunStats rs = run_once(&index, all_queries, (int)threads);
            std::cout << "  run " << (r+1) << " threads=" << threads
                      << " avg_lat_ns=" << rs.avg_latency_ns
                      << " wall_s=" << rs.wall_seconds
                      << " avg_block=" << rs.avg_block_per_lookup
                      << " avg_ic=" << rs.avg_inblock_per_lookup << "\n";
            sum_lat_ns += rs.avg_latency_ns;
            sum_wall += rs.wall_seconds;
            sum_height += rs.avg_block_per_lookup;
            sum_ios += rs.avg_inblock_per_lookup;
        }
        double avg_lat_ns = sum_lat_ns / repeats;
        double avg_wall = sum_wall / repeats;
        double avg_height = sum_height / repeats;
        double avg_ios = sum_ios / repeats;

        csv << threads << "," << avg_lat_ns << "," << avg_wall << "," << avg_height << "," << avg_ios << "\n";
        csv.flush();
        std::cout << "-> threads=" << threads << " done. avg_lat_ns=" << avg_lat_ns << " avg_ios=" << avg_ios << "\n";
    }

    csv.close();
    std::cout << "Finished. Results saved to aulid_multithread.csv\n";
    return 0;
}
