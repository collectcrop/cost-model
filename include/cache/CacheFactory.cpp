#include "cache/CacheInterface.hpp"
#include "cache/ShardedFIFOCache.hpp"
#include "cache/ShardedLRUCache.hpp"
#include "cache/ShardedLFUCache.hpp"
#include "cache/NullCache.hpp" 
#include <thread>
#include <algorithm>

namespace falcon {

static inline size_t round_up_pow2(size_t x) {
    if (x == 0) return 1;
    // 把 x 向上凑成 2^k
    --x;
    x |= x >> 1;  x |= x >> 2;  x |= x >> 4;
    x |= x >> 8;  x |= x >> 16; x |= x >> 32;
    return x + 1;
}

std::unique_ptr<ICache>
MakeShardedCache(CachePolicy policy,
                 size_t memory_budget_bytes,
                 size_t page_size,
                 size_t shards)
{
    if (page_size == 0) page_size = falcon::PAGE_SIZE;
    size_t cap_pages = memory_budget_bytes / page_size;

    // if (shards == 0) {
    //     size_t hc = std::max(8u, 2 * std::thread::hardware_concurrency());
    //     shards = round_up_pow2(hc);
    // }

    switch (policy) {
        case CachePolicy::FIFO: return std::make_unique<ShardedFIFOCache>(cap_pages, shards);
        case CachePolicy::LRU:  return std::make_unique<ShardedLRUCache>(cap_pages, shards);
        case CachePolicy::LFU:  return std::make_unique<ShardedLFUCache>(cap_pages, shards);
        case CachePolicy::NONE: return std::make_unique<NullCache>();
        default:                return std::make_unique<ShardedLRUCache>(cap_pages, shards);
    }
}

} // namespace pgm
