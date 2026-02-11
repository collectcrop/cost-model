#pragma once
#include <vector>
#include <unordered_map>
#include <list>
#include <mutex>
#include <limits>
#include <memory>
#include <string>
#include <cassert>
#include "FALCON/cache/CacheInterface.hpp"  // 按你的树调整路径

namespace falcon {

struct LFUShard {
    struct Node { size_t key; Page page; uint32_t freq; };
    // freq -> 双向链表（同频次 LRU）
    std::unordered_map<uint32_t, std::list<Node>> f2list;
    // key -> (freq, iterator)
    struct Loc { uint32_t freq; std::list<Node>::iterator it; };
    std::unordered_map<size_t, Loc> map;

    std::mutex m;
    uint32_t minFreq{0};
    size_t capacity{0};

    LFUShard() = default;
    explicit LFUShard(size_t cap) : capacity(cap) {}
    LFUShard(const LFUShard&) = delete;
    LFUShard& operator=(const LFUShard&) = delete;
};

class ShardedLFUCache final : public ICache {
public:
    ShardedLFUCache(size_t cap_pages, size_t shards)
      : total_cap_(cap_pages),
        mask_(shards - 1)
    {
        assert(shards && ((shards & (shards - 1)) == 0) && "shards must be power of two");
        shards_vec_.reserve(shards);
        size_t per = std::max<size_t>(1, cap_pages / shards);
        size_t rem = cap_pages % shards;
        for (size_t i = 0; i < shards; ++i) {
            size_t cap = per + (i < rem ? 1 : 0);
            shards_vec_.emplace_back(std::make_unique<LFUShard>(cap));
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
        // 提升频次：从 old freq list 移到 new freq list 的 front（MRU）
        auto loc = it->second;
        auto& old_list = sh.f2list[loc.freq];
        auto node = std::move(*loc.it);
        old_list.erase(loc.it);
        if (old_list.empty()) {
            sh.f2list.erase(loc.freq);
            if (sh.minFreq == loc.freq) sh.minFreq++;
        }
        node.freq++;
        auto& new_list = sh.f2list[node.freq];
        new_list.push_front(std::move(node));
        it->second = { new_list.front().freq, new_list.begin() };

        out = new_list.begin()->page; // 共享句柄拷贝
        stats_.hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void put(size_t pageIndex, Page&& page) override {
        auto& sh = *shard_of(pageIndex);
        std::lock_guard<std::mutex> lg(sh.m);
        auto it = sh.map.find(pageIndex);
        if (it != sh.map.end()) {
            // 更新值，并提升为 MRU（freq+1）
            auto loc = it->second;
            auto& old_list = sh.f2list[loc.freq];
            auto node = std::move(*loc.it);
            old_list.erase(loc.it);
            if (old_list.empty()) {
                sh.f2list.erase(loc.freq);
                if (sh.minFreq == loc.freq) sh.minFreq++;
            }
            node.page = std::move(page);
            node.freq++;
            auto& new_list = sh.f2list[node.freq];
            new_list.push_front(std::move(node));
            it->second = { new_list.front().freq, new_list.begin() };
            stats_.puts.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // 逐出（按 minFreq 的 LRU）
        while (current_size_unlocked(sh) >= sh.capacity && sh.capacity > 0) {
            auto fit = sh.f2list.find(sh.minFreq);
            // 找到 minFreq 对应的非空链表
            while (fit == sh.f2list.end() || fit->second.empty()) {
                ++sh.minFreq;
                fit = sh.f2list.find(sh.minFreq);
                if (sh.minFreq == std::numeric_limits<uint32_t>::max()) break;
            }
            if (fit == sh.f2list.end() || fit->second.empty()) break; // 防御
            auto& lst = fit->second;
            auto victim_it = std::prev(lst.end()); // LRU
            size_t victim_key = victim_it->key;
            lst.erase(victim_it);
            sh.map.erase(victim_key);
            if (lst.empty()) sh.f2list.erase(sh.minFreq);
            stats_.evictions.fetch_add(1, std::memory_order_relaxed);
        }

        // 插入新项，freq=1
        auto& lst = sh.f2list[1];
        lst.push_front(LFUShard::Node{pageIndex, std::move(page), 1});
        sh.map[pageIndex] = {1, lst.begin()};
        sh.minFreq = 1;
        stats_.puts.fetch_add(1, std::memory_order_relaxed);
    }

    void clear() override {
        for (auto& up : shards_vec_) {
            auto& sh = *up;
            std::lock_guard<std::mutex> lg(sh.m);
            sh.f2list.clear();
            sh.map.clear();
            sh.minFreq = 0;
        }
    }

    size_t size_pages() const override {
        size_t s = 0;
        for (auto& up : shards_vec_) {
            auto& sh = *up;
            std::lock_guard<std::mutex> lg(sh.m);
            s += current_size_unlocked(sh);
        }
        return s;
    }

    size_t capacity_pages() const override { return total_cap_; }
    const CacheStats& stats() const override { return stats_; }
    std::string name() const override { return "ShardedLFU"; }

private:
    // 估算当前 shard 占用页数 = 所有 freq 列表长度之和
    static size_t current_size_unlocked(const LFUShard& sh) {
        size_t s = 0;
        for (auto& kv : sh.f2list) s += kv.second.size();
        return s;
    }

    inline std::unique_ptr<LFUShard>& shard_of(size_t pageIndex) {
        return shards_vec_[pageIndex & mask_];
    }
    inline const std::unique_ptr<LFUShard>& shard_of(size_t pageIndex) const {
        return shards_vec_[pageIndex & mask_];
    }

    size_t total_cap_;
    std::vector<std::unique_ptr<LFUShard>> shards_vec_; // 关键：unique_ptr 避免移动 Shard 本体
    size_t mask_;
    mutable CacheStats stats_;
};

} // namespace pgm
