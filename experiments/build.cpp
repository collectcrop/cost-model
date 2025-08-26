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

const bool binary_search_record(pgm::Record* records, size_t lo, size_t hi, KeyType target_key){
    size_t left = lo;
    size_t right = hi;
    while (left <= right) {
        size_t mid = (right - left) / 2;
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
template <size_t Epsilon>
void benchmark(std::vector<KeyType> data,std::string filename) {
    pgm::PGMIndex<KeyType, Epsilon, 20, pgm::CacheType::DATA, Epsilon> index(data,filename,pgm::CacheStrategy::LRU);
    auto height = index.height();
    auto size = index.size_in_bytes();
    std::cout << "epsilon: " << Epsilon << " height: " << height << " index size: " << size << std::endl;
}

int main() {
    std::string filename = "wiki_ts_100M_uint64_unique";
    // std::string query_filename = "range_query_fb_uu.bin";
    std::string file = DATASETS + filename;
    // std::string query_file = DATASETS + query_filename;
    std::vector<KeyType> data = load_data(file);
    // std::vector<RangeQuery> queries = load_queries(query_file);
    const size_t MemoryBudget = 80*1024*1024;

    for (size_t epsilon : {4,8,16,32,64,128,256}){
        switch (epsilon) {
            case 4: 
                benchmark<4>(data,file);
                break;
            case 8:
                benchmark<8>(data,file);
                break;
            case 16:
                benchmark<16>(data,file);
                break;
            case 32:
                benchmark<32>(data,file);
                break;
            case 64:
                benchmark<64>(data,file);
                break;
            case 128:
                benchmark<128>(data,file);
                break;
            case 256:
                benchmark<256>(data,file);
                break;
        }
    }
    
    return 0;
}


