#pragma once
#include <vector>
#include <unordered_map>
#include <deque>
#include <mutex>
#include <memory>
#include <string>
#include <cassert>
#include "cache/CacheInterface.hpp"  // 按你的树调整路径

namespace falcon {

struct FIFOShard {
    struct Entry { size_t key; Page page; };

    std::mutex m;
    std::unordered_map<size_t, std::deque<Entry>::iterator> map;
    std::deque<Entry> q;   // FIFO 队列
    size_t capacity{0};

    FIFOShard() = default;
    explicit FIFOShard(size_t cap) : capacity(cap) {}
    FIFOShard(const FIFOShard&) = delete;
    FIFOShard& operator=(const FIFOShard&) = delete;
};

class ShardedFIFOCache final : public ICache {
public:
    ShardedFIFOCache(size_t cap_pages, size_t shards)
      : total_cap_(cap_pages),
        mask_(shards - 1)
    {
        assert(shards && ((shards & (shards - 1)) == 0) && "shards must be power of two");
        shards_vec_.reserve(shards);
        // 均分容量，至少 1 页/分片（避免 0）
        size_t per = std::max<size_t>(1, cap_pages / shards);
        size_t rem = cap_pages % shards;
        for (size_t i = 0; i < shards; ++i) {
            size_t cap = per + (i < rem ? 1 : 0);
            shards_vec_.emplace_back(std::make_unique<FIFOShard>(cap));
        }
    }

    bool get(size_t pageIndex, Page& out) override {
        auto& sh = *shard_of(pageIndex);
        std::lock_guard<std::mutex> lg(sh.m);
        auto it = sh.map.find(pageIndex);
        if (it == sh.map.end()) {
            stats_.misses.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        out = it->second->page; // 共享句柄拷贝（Page 里是 shared_ptr）
        stats_.hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void put(size_t pageIndex, Page&& page) override {
        auto& sh = *shard_of(pageIndex);
        std::lock_guard<std::mutex> lg(sh.m);

        auto it = sh.map.find(pageIndex);
        if (it != sh.map.end()) {
            // 覆盖旧值，保持 FIFO 位置不变
            it->second->page = std::move(page);
            stats_.puts.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // 逐出直到容量足够
        while (sh.q.size() >= sh.capacity && !sh.q.empty()) {
            auto& front = sh.q.front();
            sh.map.erase(front.key);
            sh.q.pop_front();
            stats_.evictions.fetch_add(1, std::memory_order_relaxed);
        }

        sh.q.push_back(FIFOShard::Entry{pageIndex, std::move(page)});
        sh.map[pageIndex] = std::prev(sh.q.end());
        stats_.puts.fetch_add(1, std::memory_order_relaxed);
    }

    void clear() override {
        for (auto& up : shards_vec_) {
            auto& sh = *up;
            std::lock_guard<std::mutex> lg(sh.m);
            sh.map.clear();
            sh.q.clear();
        }
    }

    size_t size_pages() const override {
        size_t s = 0;
        for (auto& up : shards_vec_) {
            auto& sh = *up;
            std::lock_guard<std::mutex> lg(sh.m);
            s += sh.q.size();
        }
        return s;
    }

    size_t capacity_pages() const override { return total_cap_; }
    const CacheStats& stats() const override { return stats_; }
    std::string name() const override { return "ShardedFIFO"; }

private:
    inline std::unique_ptr<FIFOShard>& shard_of(size_t pageIndex) {
        return shards_vec_[pageIndex & mask_];
    }
    inline const std::unique_ptr<FIFOShard>& shard_of(size_t pageIndex) const {
        return shards_vec_[pageIndex & mask_];
    }

    size_t total_cap_;
    std::vector<std::unique_ptr<FIFOShard>> shards_vec_; // 关键：unique_ptr 避免移动 Shard 本体
    size_t mask_;                                        // shards-1
    mutable CacheStats stats_;
};

} // namespace pgm
