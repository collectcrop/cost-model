#pragma once
#include <string>
#include <atomic>
#include "cache/CacheInterface.hpp"   // 或你的实际路径

namespace pgm {

class NullCache final : public ICache {
public:
    NullCache() = default;

    // 永远 miss；为了统计，累计 misses
    bool get(size_t /*pageIndex*/, Page& /*out*/) override {
        stats_.misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    // 不存任何内容；这里不增加 puts 计数，避免干扰
    void put(size_t /*pageIndex*/, Page&& /*page*/) override {}

    void clear() override {}

    size_t size_pages() const override { return 0; }
    size_t capacity_pages() const override { return 0; }
    const CacheStats& stats() const override { return stats_; }
    std::string name() const override { return "NullCache"; }

private:
    mutable CacheStats stats_{};
};

} // namespace pgm
