#pragma once
#include <list>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <cassert>
#include "FALCON/cache/CacheInterface.hpp"  

namespace falcon {

struct LRUShard {
    std::list<size_t> lru;
    struct Entry { Page page; std::list<size_t>::iterator it; };
    std::unordered_map<size_t, Entry> map;
    mutable std::shared_mutex mtx;

    size_t capacity = 0; 

    LRUShard() = default;
    explicit LRUShard(size_t cap) : capacity(cap) {}
    LRUShard(const LRUShard&) = delete;
    LRUShard& operator=(const LRUShard&) = delete;
};

class ShardedLRUCache final : public ICache {
public:
    ShardedLRUCache(size_t cap_pages, size_t shards)
      : total_cap_(cap_pages),
        mask_(shards - 1)
    {
        assert((shards & (shards - 1)) == 0 && "shards must be power of two");
        shards_vec_.reserve(shards);
        size_t per = std::max<size_t>(1, cap_pages / shards);
        for (size_t i = 0; i < shards; ++i)
            shards_vec_.emplace_back(std::make_unique<LRUShard>(per));
    }

    bool get(size_t pageIndex, Page& out) override {
        auto& sh = *shard_of(pageIndex);
        {   // 读路径：先尝试共享锁
            std::shared_lock rlk(sh.mtx);
            auto it = sh.map.find(pageIndex);
            if (it != sh.map.end()) {
                stats_.hits.fetch_add(1, std::memory_order_relaxed);
                out = it->second.page; // 共享底层 buffer（Page 里是 shared_ptr）
                // 升级为写锁做 LRU 调整（简单起见：释放读锁->加写锁）
                rlk.unlock();
                std::unique_lock wlk(sh.mtx);
                sh.lru.erase(it->second.it);
                sh.lru.push_front(pageIndex);
                it->second.it = sh.lru.begin();
                return true;
            }
        }
        stats_.misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    void put(size_t pageIndex, Page&& page) override {
        auto& sh = *shard_of(pageIndex);
        std::unique_lock wlk(sh.mtx);
        auto it = sh.map.find(pageIndex);
        if (it != sh.map.end()) {
            it->second.page = std::move(page);
            sh.lru.erase(it->second.it);
            sh.lru.push_front(pageIndex);
            it->second.it = sh.lru.begin();
        } else {
            if (sh.map.size() >= sh.capacity) {
                // 逐出
                auto victim = sh.lru.back();
                sh.lru.pop_back();
                sh.map.erase(victim);
                stats_.evictions.fetch_add(1, std::memory_order_relaxed);
            }
            sh.lru.push_front(pageIndex);
            typename LRUShard::Entry e{ std::move(page), sh.lru.begin() };
            sh.map.emplace(pageIndex, std::move(e));
        }
        stats_.puts.fetch_add(1, std::memory_order_relaxed);
    }

    void clear() override {
        for (auto& p : shards_vec_) {
            std::unique_lock wlk(p->mtx);
            p->lru.clear();
            p->map.clear();
        }
    }

    size_t size_pages() const override {
        size_t s = 0;
        for (auto& p : shards_vec_) {
            std::shared_lock rlk(p->mtx);
            s += p->map.size();
        }
        return s;
    }

    size_t capacity_pages() const override { return total_cap_; }
    const CacheStats& stats() const override { return stats_; }
    std::string name() const override { return "ShardedLRU"; }

private:
    inline std::unique_ptr<LRUShard>& shard_of(size_t pageIndex) {
        return shards_vec_[(pageIndex) & mask_]; // shards 为 2^k，可用位与
    }
    inline const std::unique_ptr<LRUShard>& shard_of(size_t pageIndex) const {
        return shards_vec_[(pageIndex) & mask_];
    }

    size_t total_cap_;
    // 重要：用 unique_ptr 装 shard，避免 vector 移动/复制
    std::vector<std::unique_ptr<LRUShard>> shards_vec_;
    size_t mask_;     // shards-1

    mutable CacheStats stats_;
};

} // namespace pgm
