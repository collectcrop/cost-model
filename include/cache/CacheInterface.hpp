#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <atomic>
#include <string>
#include <vector>

#include "utils/include.hpp"   // 里有 Page / IOResult / PAGE_SIZE 等

namespace falcon {


struct CacheStats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> puts{0};
    std::atomic<uint64_t> evictions{0};
};

class ICache {
public:
    virtual ~ICache() = default;

    // 命中返回 true，并把页面句柄写入 out（共享底层内存）；未命中返回 false
    virtual bool get(size_t pageIndex, Page& out) = 0;

    // 插入/更新；必要时在当前 shard 内逐出
    virtual void put(size_t pageIndex, Page&& page) = 0;

    // 清空
    virtual void clear() = 0;

    // 统计与容量
    virtual size_t size_pages() const = 0;          // 总页数（所有 shard 之和）
    virtual size_t capacity_pages() const = 0;      // 总容量（页）
    virtual const CacheStats& stats() const = 0;

    // 诊断：返回实现名称
    virtual std::string name() const = 0;
};

// 只做声明，定义放到 CacheFactory.cpp 里
std::unique_ptr<ICache>
MakeShardedCache(CachePolicy policy,
                 size_t memory_budget_bytes,
                 size_t page_size,
                 size_t shards);

} // namespace pgm
