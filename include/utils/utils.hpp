#pragma once
#include <vector>
#include <cstring>
#include <cstdint>
#include "include.hpp"

// 定义 KeyType（要和主程序保持一致）
using KeyType = uint64_t;

// 加载数据
std::vector<KeyType> load_data(const std::string& filename, size_t total_keys) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Failed to open input file: " << filename << "\n";
        return {};
    }

    std::vector<KeyType> keys(total_keys);
    infile.read(reinterpret_cast<char*>(keys.data()), total_keys * sizeof(KeyType));

    if (!infile) {
        std::cerr << "Error while reading input file data.\n";
    }
    return keys;
}

// 加载查询
std::vector<KeyType> load_queries(const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary);
    infile.seekg(0, std::ios::end);
    size_t filesize = infile.tellg();
    infile.seekg(0);
    size_t num_queries = filesize / sizeof(KeyType);
    std::vector<KeyType> queries(num_queries);
    infile.read(reinterpret_cast<char*>(queries.data()), filesize);
    return queries;
}

// 提取 key
uint64_t extract_key(const char* record) {
    uint64_t key;
    std::memcpy(&key, record, sizeof(uint64_t));
    return key;
}

// 二分搜索 buffer
// const char* binary_search_record(const std::vector<char>& buffer, size_t lo, size_t hi, uint64_t target_key) {
//     size_t left = lo;
//     size_t right = hi;

//     while (left < right) {
//         size_t mid = left + (right - left) / 2;
//         const char* mid_ptr = buffer.data() + mid * sizeof(KeyType); // 注意这里 RECORD_SIZE
//         uint64_t mid_key = extract_key(mid_ptr);

//         if (mid_key < target_key) {
//             left = mid + 1;
//         } else {
//             right = mid;
//         }
//     }

//     if (left < hi) {
//         const char* candidate = buffer.data() + left * sizeof(KeyType);
//         uint64_t candidate_key = extract_key(candidate);
//         if (candidate_key == target_key) {
//             return candidate;
//         }
//     }
//     return nullptr;
// }

bool binary_search_record(pgm::Record* records, size_t lo, size_t hi, KeyType target_key){
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
