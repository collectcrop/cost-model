// LatencyRecorder.hpp
#pragma once
#include <vector>
#include <chrono>
#include <cstdint>
#include <string>
#include <fstream>
#include <type_traits>
#include <cmath>

class LatencyRecorder {
public:
    explicit LatencyRecorder(size_t total_queries)
        : lat_ns_(total_queries, 0) {}

    // 直接设置（end - start 已在外部算好）
    inline void set(size_t qid, uint64_t ns) noexcept {
        lat_ns_[qid] = ns;
    }

    inline double get_avg() const noexcept {
        uint64_t sum = 0;
        for (auto v : lat_ns_) sum += v;
        return double(sum) / lat_ns_.size();
    }

    uint64_t get_percentile(double p) const {
        if (lat_ns_.empty()) return 0;
        std::vector<uint64_t> tmp = lat_ns_;   
        size_t n = tmp.size();
        // p=0.99 → 取 ceil(0.99*n)-1
        size_t k = std::min(n - 1, (size_t)std::max<size_t>(0, (size_t)std::ceil(p * n) - 1));
        std::nth_element(tmp.begin(), tmp.begin() + k, tmp.end());
        return tmp[k];
    }

    uint64_t p50() const { return get_percentile(0.50); }
    uint64_t p95() const { return get_percentile(0.95); }
    uint64_t p99() const { return get_percentile(0.99); }

    // 包一层：测量一个可调用对象（lambda / 函数），并记录到 qid
    template<class Fn>
    inline auto time_call(size_t qid, Fn&& fn)
        -> decltype(fn()) {
        const auto t0 = clock_t::now();
        if constexpr (std::is_void_v<decltype(fn())>) {
            fn();
            const auto t1 = clock_t::now();
            lat_ns_[qid] = to_ns(t1 - t0);
        } else {
            auto ret = fn();
            const auto t1 = clock_t::now();
            lat_ns_[qid] = to_ns(t1 - t0);
            return ret;
        }
    }

    const std::vector<uint64_t>& data() const noexcept { return lat_ns_; }

    // 导出 CSV：query_index,key,thread_id,found,latency_ns
    template<typename KeyT>
    void dump_csv(const std::string& path,
                  const std::vector<KeyT>& queries,
                  const std::vector<uint32_t>* thread_id = nullptr,
                  const std::vector<uint8_t>* found = nullptr) const
    {
        std::ofstream out(path);
        out << "query_index,key";
        if (thread_id) out << ",thread_id";
        if (found)     out << ",found";
        out << ",latency_ns\n";

        const size_t n = lat_ns_.size();
        for (size_t i = 0; i < n; ++i) {
            out << i << "," << queries[i];
            if (thread_id) out << "," << (*thread_id)[i];
            if (found)     out << "," << static_cast<int>((*found)[i]);
            out << "," << lat_ns_[i] << "\n";
        }
    }

private:
    using clock_t = std::chrono::high_resolution_clock;

    static inline uint64_t to_ns(std::chrono::high_resolution_clock::duration d) {
        return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
    }

    std::vector<uint64_t> lat_ns_;
};
