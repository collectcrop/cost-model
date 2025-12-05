#pragma once
#include <vector>
#include <cstring>
#include <cstdint>
#include <limits>
#include <fstream>
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


// ---------- binary loaders ----------
template<typename T>
static std::vector<T> load_binary(const std::string& filename, bool has_header=false) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { throw std::runtime_error("open failed: " + filename); }
    size_t n = 0;
    if (has_header) {
        uint64_t total; in.read(reinterpret_cast<char*>(&total), sizeof(total));
        n = total;
    } else {
        in.seekg(0, std::ios::end);
        auto bytes = static_cast<size_t>(in.tellg());
        in.seekg(0, std::ios::beg);
        n = bytes / sizeof(T);
    }
    std::vector<T> v(n);
    in.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
    if (!in) throw std::runtime_error("read failed: " + filename);
    return v;
}

// static std::vector<pgm::RangeQ> load_ranges(const std::string& filename) {
//     std::ifstream in(filename, std::ios::binary);
//     if (!in) { std::cerr << "open range file failed: " << filename << "\n"; return {}; }
//     in.seekg(0, std::ios::end);
//     size_t bytes = static_cast<size_t>(in.tellg());
//     in.seekg(0, std::ios::beg);
//     if (bytes % sizeof(pgm::RangeQ) != 0) {
//         std::cerr << "range file size not multiple of 16 bytes: " << bytes << "\n";
//         return {};
//     }
//     size_t n = bytes / sizeof(pgm::RangeQ);
//     std::vector<pgm::RangeQ> rq(n);
//     in.read(reinterpret_cast<char*>(rq.data()), bytes);
//     return rq;
// }

std::vector<pgm::RangeQ> load_ranges(const std::string& filename) {
    std::vector<pgm::RangeQ> queries;
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open file " << filename << std::endl;
        return queries;
    }

    pgm::RangeQ rq;
    while (fin.read(reinterpret_cast<char*>(&rq), sizeof(pgm::RangeQ))) {
        queries.push_back(rq);
    }

    return queries;
}


// 提取 key
uint64_t extract_key(const char* record) {
    uint64_t key;
    std::memcpy(&key, record, sizeof(uint64_t));
    return key;
}

bool binary_search_record(pgm::Record* records, size_t lo, size_t hi, KeyType target_key){
    size_t left = lo;
    size_t right = hi;
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

template<typename T>
std::vector<T> load_data_pgm_safe(const std::string &filename, size_t n) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open data file: " + filename);
    }

    std::vector<T> data;
    data.reserve(n);

    constexpr T SENTINEL = std::numeric_limits<T>::max();
    T x;
    size_t cnt = 0, fixed = 0;

    while (cnt < n && in.read(reinterpret_cast<char*>(&x), sizeof(T))) {
        if (x == SENTINEL) {
            ++fixed;
            x = SENTINEL - 1;      
        }
        data.push_back(x);
        ++cnt;
    }

    std::cerr << "[load_data_pgm_safe] loaded=" << data.size()
              << ", sentinel_fixed=" << fixed << std::endl;

    return data;
}

std::vector<pgm::RangeQ> load_ranges_pgm_safe(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open range file: " + filename);
    }

    std::vector<pgm::RangeQ> ranges;
    constexpr KeyType SENTINEL = std::numeric_limits<KeyType>::max();

    pgm::RangeQ q;
    size_t total = 0;
    size_t skipped_sentinel = 0;

    while (in.read(reinterpret_cast<char*>(&q), sizeof(pgm::RangeQ))) {
        ++total;

        // 跳过包含 SENTINEL 的区间（lo 或 hi 任一为哨兵值）
        if (q.lo == SENTINEL || q.hi == SENTINEL) {
            ++skipped_sentinel;
            continue;
        }

        //（可选）保证 lo <= hi，如果你的生成脚本已经保证这一点，可以删掉
        if (q.lo > q.hi) {
            std::swap(q.lo, q.hi);
        }

        ranges.push_back(q);
    }

    std::cerr << "[load_ranges_pgm_safe] total_ranges=" << total
              << ", kept=" << ranges.size()
              << ", skipped_sentinel=" << skipped_sentinel
              << std::endl;

    return ranges;
}

template<typename T>
std::vector<T> load_queries_pgm_safe(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open query file: " + filename);
    }

    std::vector<T> qs;
    constexpr T SENTINEL = std::numeric_limits<T>::max();

    T x;
    size_t fixed = 0;
    while (in.read(reinterpret_cast<char*>(&x), sizeof(T))) {
        if (x == SENTINEL) {
            ++fixed;
            x = SENTINEL - 1;
        }
        qs.push_back(x);
    }

    std::cerr << "[load_queries_pgm_safe] loaded_queries=" << qs.size()
              << ", sentinel_fixed=" << fixed << std::endl;

    return qs;
}