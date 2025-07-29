/*
 * This example shows how to index and query a vector of random integers with the PGM-index.
 * Compile with:
 *   g++ simple.cpp -std=c++17 -I../include -o simple
 * Run with:
 *   ./simple
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <fstream>
#include "distribution/zipf.hpp"
#include "pgm/pgm_index_cost.hpp"

using KeyType = uint64_t;

struct Record {
    uint64_t key;
    uint64_t data;
};
struct BenchmarkResult {
    size_t epsilon;
    double time_ns;
    double hit_ratio;
};

using timer = std::chrono::high_resolution_clock;

std::vector<KeyType> generate_data(const std::string& filename, size_t num_records) {
    std::ofstream ofs(filename, std::ios::binary);
    std::mt19937 rng(42);
    std::uniform_int_distribution<KeyType> key_dist(1, 1e9);

    std::vector<Record> records;
    for (size_t i = 0; i < num_records; ++i) {
        Record r;
        r.key = key_dist(rng);
        r.data = std::rand(); 
        records.push_back(r);
    }
    records.push_back({66666666,7777});
    std::sort(records.begin(), records.end(), [](const Record& a, const Record& b) {
        return a.key < b.key;
    });

    for (size_t i = 0; i < records.size(); ++i) {
        ofs.write(reinterpret_cast<const char*>(&records[i]), sizeof(Record));
    }

    ofs.close();

    // extract keys
    std::vector<uint64_t> keys;
    keys.reserve(records.size());
    for (const auto& record : records) {
        keys.push_back(record.key);
    }

    return keys;  
}

uint64_t extract_key(const char* record) {
    // 假设 key 是前 8 个字节（little-endian）
    uint64_t key;
    std::memcpy(&key, record, sizeof(uint64_t));
    return key;
}

const char* binary_search_record(const std::vector<char>& buffer, size_t lo, size_t hi, uint64_t target_key) {
    size_t left = lo;
    size_t right = hi;

    while (left < right) {
        size_t mid = left + (right - left) / 2;
        const char* mid_ptr = buffer.data() + mid * RECORD_SIZE;
        uint64_t mid_key = extract_key(mid_ptr);

        if (mid_key < target_key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    // 检查 left 是否就是目标
    if (left < hi) {
        const char* candidate = buffer.data() + left * RECORD_SIZE;
        uint64_t candidate_key = extract_key(candidate);
        if (candidate_key == target_key) {
            return candidate; // 找到
        }
    }

    return nullptr; // 没找到
}

std::vector<KeyType>
generate_queries(KeyType minKey, KeyType maxKey, size_t num_queries = 10000000) {
    size_t n = maxKey - minKey;
    std::uniform_int_distribution<KeyType> distribution(0, n - 1);
    std::mt19937 generator(std::random_device{}());
    std::vector<KeyType> queries;
    queries.reserve(num_queries);

    for (size_t i = 0; i < num_queries; ++i)
        queries.push_back(distribution(generator));

    std::shuffle(queries.begin(), queries.end(), generator);
    return queries;
}

std::vector<KeyType>
generate_zipf_queries(KeyType minKey, KeyType maxKey, double alpha = 1.0, size_t num_queries = 10000000) {
    std::mt19937 rng(std::random_device{}());
    zipf_distribution<KeyType, double> zipf(maxKey - minKey, alpha);

    std::vector<KeyType> queries;
    queries.reserve(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        KeyType k = zipf(rng);              // k ∈ [1, N]
        queries.push_back(minKey + k - 1);  // 映射回 [minKey, maxKey)
    }
    return queries;
}

template <size_t Epsilon, size_t M>
BenchmarkResult benchmark(std::vector<KeyType> data,std::vector<KeyType> queries) {
    pgm::PGMIndex<KeyType, Epsilon, M> index(data);
    auto t0 = timer::now();
    for (auto &q : queries) {
        auto range = index.search(q);
        std::vector<char> buffer = range.buffer;
        size_t lo = range.lo;
        size_t hi = range.hi;
        const char* result = binary_search_record(buffer, lo, hi, q);
    }
    auto t1 = timer::now();
    auto query_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / queries.size();
    // total_time += query_ns;
    size_t total_hits = index.get_cache_hits();
    size_t total_access = index.get_cache_hits() + index.get_cache_misses();
    
    BenchmarkResult result;
    result.epsilon = Epsilon;
    result.time_ns = query_ns;
    result.hit_ratio = static_cast<double>(total_hits) / total_access;

    return result;
}

void single_test(){
    size_t num_records = 1000000;

    std::vector<KeyType> data = generate_data(DATA_FILE, num_records);
    // std::vector<KeyType> queries = generate_zipf_queries(data.front(), data.back(),1);
    const size_t MemoryBudget = 1<<5;
    KeyType q = 66666666;
    pgm::PGMIndex<KeyType, 32, MemoryBudget> index(data);
    auto range = index.search(q);
    std::vector<char> buffer = range.buffer;
    size_t lo = range.lo;
    size_t hi = range.hi;
    const char* result = binary_search_record(buffer, lo, hi, q);
    if (result) {
        uint64_t key, value;
        std::memcpy(&key, result, sizeof(uint64_t));
        std::memcpy(&value, result + sizeof(uint64_t), sizeof(uint64_t));
        std::cout << "Found key=" << key << ", value=" << value << std::endl;
    } else {
        std::cout << "Key " << q << " not found." << std::endl;
    }
    std::cout << index.get_segment_size() << std::endl;
}

int main() {
    size_t num_records = 1000000;

    std::vector<KeyType> data = generate_data(DATA_FILE, num_records);
    // std::vector<KeyType> queries = generate_zipf_queries(data.front(), data.back(),1);
    const size_t MemoryBudget = 1<<22;
    
    int trials = 1;
    std::ofstream ofs("benchmark_results.csv");
    ofs << "epsilon,avg_query_time_ns,avg_cache_hit_ratio\n";

    for (int t=0; t<trials; t++){
        for (size_t epsilon : {2,4,8,9}) {
            BenchmarkResult result;
            switch (epsilon) {
                case 2: result = benchmark<2, MemoryBudget>(data, queries); break;
                case 4: result = benchmark<4, MemoryBudget>(data, queries); break;
                case 8:  result = benchmark<8, MemoryBudget>(data, queries); break;
                case 9: result = benchmark<9, MemoryBudget>(data, queries); break;
                case 10: result = benchmark<10, MemoryBudget>(data, queries); break;
                case 11: result = benchmark<11, MemoryBudget>(data, queries); break;
                case 12: result = benchmark<12, MemoryBudget>(data, queries); break;
                case 13: result = benchmark<13, MemoryBudget>(data, queries); break;
                case 14: result = benchmark<14, MemoryBudget>(data, queries); break;
                case 15: result = benchmark<15, MemoryBudget>(data, queries); break;
                case 16: result = benchmark<16, MemoryBudget>(data, queries); break;
                case 17: result = benchmark<17, MemoryBudget>(data, queries); break;
                case 18: result = benchmark<18, MemoryBudget>(data, queries); break;
                case 19: result = benchmark<19, MemoryBudget>(data, queries); break;
                case 20: result = benchmark<20, MemoryBudget>(data, queries); break;
                case 21: result = benchmark<21, MemoryBudget>(data, queries); break;
                case 22: result = benchmark<22, MemoryBudget>(data, queries); break;
                case 23: result = benchmark<23, MemoryBudget>(data, queries); break;
                case 24: result = benchmark<24, MemoryBudget>(data, queries); break;
                case 25: result = benchmark<25, MemoryBudget>(data, queries); break;
                case 26: result = benchmark<26, MemoryBudget>(data, queries); break;
                case 32: result = benchmark<32, MemoryBudget>(data, queries); break;
                case 64: result = benchmark<64, MemoryBudget>(data, queries); break;
                case 128: result = benchmark<128, MemoryBudget>(data, queries); break;
            }

            std::cout << "ε=" << result.epsilon
                    << ", time=" << result.time_ns << " ns"
                    << ", hit ratio=" << result.hit_ratio << std::endl;
            ofs << result.epsilon << "," << result.time_ns << "," << result.hit_ratio << "\n";
        }
    }
    
    ofs.close();
    
    return 0;
}


