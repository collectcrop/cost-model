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
#define DIRECTORY "/mnt/home/zwshi/learned-index/cost-model/experiments/"
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"

struct Record {
    uint64_t key;
};

struct RangeQuery {
    uint64_t lo;
    uint64_t hi;
};

using timer = std::chrono::high_resolution_clock;

std::vector<KeyType> load_data(std::string filename){
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Failed to open input file: " << filename << "\n";
        return {};
    }
    // Read total number of keys in the original file
    uint64_t total_keys;
    infile.read(reinterpret_cast<char*>(&total_keys), sizeof(uint64_t));

    std::vector<KeyType> keys(total_keys);
    infile.read(reinterpret_cast<char*>(keys.data()), total_keys * sizeof(Record));

    if (!infile) {
        std::cerr << "Error while reading input file data.\n";
    }

    return keys;
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

const bool binary_search_record(std::vector<pgm::Record> records, size_t lo, size_t hi, KeyType target_key){
    int64_t left = lo;
    int64_t right = hi;
    while (left <= right) {
        int64_t mid = (right + left) / 2;
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


int main() {
    std::string filename = "fb_20M_uint64_unique";
    // std::string query_filename = "range_query_fb_uu.bin";
    std::string file = DATASETS + filename;
    // std::string query_file = DATASETS + query_filename;
    std::vector<KeyType> data = load_data(file);
    // std::vector<RangeQuery> queries = load_queries(query_file);
    const size_t MemoryBudget = 80*1024*1024;
    size_t query = 112983;
    pgm::PGMIndex<KeyType, 64, MemoryBudget, pgm::CacheType::DATA> index(data,file,pgm::CacheStrategy::LRU);
    // std::vector<KeyType> res = index.range_search(5237953133,5255844371);
    auto range = index.search(query,pgm::ALL_IN_ONCE);
    std::vector<pgm::Record> records = range.records;
    size_t lo = range.lo;
    size_t hi = range.hi;
    bool result = binary_search_record(records,lo,hi,query);
    if (result){
        std::cout << "found key " << query << std::endl;
    }else{
        std::cout << "not found" << std::endl;
    }
    // const char* result = binary_search_record(buffer, lo, hi, query);
    // if (result==nullptr){
    //     std::cout << "not found" << std::endl;
    // }else{
    //     std::cout << "found key " << query << std::endl;
    // }
    // for (auto &it:res){
    //     std::cout << it << ",";
    // }
    // std::cout<<std::endl<<res.size()<<std::endl;
    
    return 0;
}


