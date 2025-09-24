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
#include "utils/include.hpp"

using KeyType = uint64_t;
#define DIRECTORY "/mnt/home/zwshi/learned-index/cost-model/experiments/"
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"

struct Record {
    uint64_t key;
};

struct RangeQuery {
    uint64_t lo;
    uint64_t hi;
};

struct BenchmarkResult {
    size_t epsilon;
    double time_ns;
    double hit_ratio;
    double index_hit_ratio;
    time_t total_time;
    time_t data_IO_time;
    time_t index_IO_time;
    size_t height;
    size_t data_IOs;
};

using timer = std::chrono::high_resolution_clock;

template<typename T>
std::vector<T> load_binary(const std::string& filename, bool has_header = false) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t num_items = 0;
    if (has_header) {
        uint64_t total;
        infile.read(reinterpret_cast<char*>(&total), sizeof(total));
        num_items = total;
    } else {
        infile.seekg(0, std::ios::end);
        size_t filesize = infile.tellg();
        infile.seekg(0);
        num_items = filesize / sizeof(T);
    }

    std::vector<T> data(num_items);
    infile.read(reinterpret_cast<char*>(data.data()), num_items * sizeof(T));

    if (!infile) {
        throw std::runtime_error("Error while reading file data.");
    }

    return data;
}

std::vector<RangeQuery> load_queries(const std::string& filename) {
    std::vector<RangeQuery> queries;
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open file " << filename << std::endl;
        return queries;
    }

    RangeQuery rq;
    while (fin.read(reinterpret_cast<char*>(&rq), sizeof(RangeQuery))) {
        queries.push_back(rq);
    }

    return queries;
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
        const char* mid_ptr = buffer.data() + mid * pgm::RECORD_SIZE;
        uint64_t mid_key = extract_key(mid_ptr);

        if (mid_key < target_key) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    // 检查 left 是否就是目标
    if (left < hi) {
        const char* candidate = buffer.data() + left * pgm::RECORD_SIZE;
        uint64_t candidate_key = extract_key(candidate);
        if (candidate_key == target_key) {
            return candidate; // 找到
        }
    }

    return nullptr; // 没找到
}
const bool binary_search_record(pgm::Record* records, size_t lo, size_t hi, KeyType target_key){
    size_t left = lo/pgm::ITEM_PER_PAGE;
    size_t right = hi/pgm::ITEM_PER_PAGE;
    while (left <= right) {
        size_t mid = (right + left) / 2;
        KeyType key = records[mid].key;
        if (key < target_key) {
            left = mid + 1;
        } else if(key > target_key){
            right = mid - 1;
        } else {
            return true;
        }
    }
    return false;
}

template <size_t Epsilon, size_t M>
BenchmarkResult benchmark(std::vector<KeyType> data,std::vector<RangeQuery> queries,std::string filename,
    pgm::CacheStrategy s, std::vector<KeyType> raw_queries={}) {
    pgm::PGMIndex<KeyType, Epsilon, M, pgm::CacheType::DATA> index(data,filename,s);
    auto t0 = timer::now();
    int cnt = 0;
    std::vector<KeyType> range;
    for (auto &q : queries) {
        range = index.range_search(q.lo,q.hi,raw_queries, pgm::ALL_IN_ONCE);
    }
    auto t1 = timer::now();
    std::cout << "total: " << range.size() << std::endl;
    auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    auto query_ns = t / queries.size();
    
    auto cache = index.get_data_cache();
    size_t total_hits = cache->get_hit_count();
    size_t total_access = cache->get_hit_count() + cache->get_miss_count();
    
    std::cout << "total access: " << total_access << ", total hits: " << total_hits << std::endl;
    std::cout << "C=" << cache->get_C() << std::endl; 
    BenchmarkResult result;
    result.epsilon = Epsilon;
    result.time_ns = query_ns;
    result.hit_ratio = static_cast<double>(total_hits) / total_access;
    result.total_time = t;
    result.data_IO_time = cache->get_IO_time();
    result.height = index.height();
    result.data_IOs = cache->get_IOs();
    return result;
}

std::string mk_outfilename(pgm::CacheStrategy s,std::string dataset,size_t ds, size_t ms, std::string suffix=""){
    std::string prefix;
    switch (s){
        case pgm::CacheStrategy::LRU:
            prefix = "LRU-";
            break;
        case pgm::CacheStrategy::FIFO:
            prefix = "FIFO-";
            break;
        case pgm::CacheStrategy::LFU:
            prefix = "LFU-";
            break;
    }
    return prefix + dataset + "-" + std::to_string(ds) + "M-M" + std::to_string(ms) + suffix +".csv";
}

int main() {
    std::string dataset = "books";
    std::string filename = "books_20M_uint64_unique";
    std::string query_filename = "books_20M_uint64_unique.100Ktable2.bin";
    std::string file = DATASETS + filename;
    std::string query_file = DATASETS + query_filename;
    std::vector<KeyType> data = load_binary<KeyType>(file,false);
    std::vector<KeyType> raw_queries = load_binary<KeyType>(query_file,false);
    std::vector<RangeQuery> queries;
    queries.push_back({raw_queries[0],raw_queries[raw_queries.size()-1]});
    const size_t MemoryBudget = 20*1024*1024;
    
    int trials = 10;    

    for (pgm::CacheStrategy s: {pgm::CacheStrategy::LRU,pgm::CacheStrategy::FIFO,pgm::CacheStrategy::LFU}){     // pgm::CacheStrategy::LRU,pgm::CacheStrategy::FIFO,pgm::CacheStrategy::LFU
        std::ofstream ofs(mk_outfilename(s,dataset,20,MemoryBudget>>20,"-range.5"));
        ofs << "epsilon,avg_query_time_ns,avg_cache_hit_ratio,data_IOs,total_time\n";
        for (size_t epsilon : {8}) {     //8,10,12,14,16,18,20,24,28,32,40,48,56,64
            BenchmarkResult result;
            for (int i=0;i<trials;i++){
                switch (epsilon) {
                    case 8:  result = benchmark<8, MemoryBudget>(data, queries, file, s, raw_queries); break;
                }
                std::cout << "ε=" << result.epsilon
                    << ", time=" << result.time_ns << " ns"
                    << ", hit ratio=" << result.hit_ratio
                    << ", total time=" << result.total_time << " ns"
                    << ", data IO time=" << result.data_IO_time << " ns"
                    << ", data IOs=" << result.data_IOs
                    << ", height=" << result.height << std::endl;
                ofs << result.epsilon << "," << result.time_ns << "," << result.hit_ratio << "," 
                <<  "," << result.data_IOs <<  "," << result.total_time <<"\n";
            }
        }
        ofs.close();
    }
    
    
    
    return 0;
}


