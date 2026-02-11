#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <cstdint>
#include <vector>
#include <atomic>
#include "FALCON/utils/include.hpp"

// global counter for completed queries (for progress tracking)
std::atomic<size_t> global_queries_done{0};

struct BaseConfig {
    std::string dataset;
    uint64_t num_keys = 0;
    int repeats = 3;
    double memory_mb = 256;
    falcon::CachePolicy policy = falcon::CachePolicy::NONE;
    std::string baseline = "FALCON";
};

struct EpsilonConfig : public BaseConfig {
    std::vector<int> epsilons;
};

struct ThreadConfig : public BaseConfig {
    std::vector<int> thread_counts;
};

struct BatchConfig : public BaseConfig {
    std::vector<int> batch_sizes;
};

struct BenchmarkResult {
    size_t epsilon;
    falcon::CachePolicy policy;
    double avg_lat;
    double hit_ratio;
    time_t total_time;
    size_t height;
    size_t data_IOs;
    double p50_lat;
    double p95_lat;
    double p99_lat;
    // time_t data_IO_time;
    // time_t index_time;
    // time_t cache_time;
};

#endif 