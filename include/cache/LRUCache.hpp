#include "CacheInterface.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>  
#include <unordered_map>
#include <memory>
#include <list>
#include <iostream>
#include <chrono>
#include <shared_mutex>
#include <mutex>

using timer = std::chrono::high_resolution_clock;


class LRUCache : public ICache<size_t, pgm::Page> {
    using LRUIter = std::list<size_t>::iterator;

    struct CacheEntry {
        pgm::Page page;
        LRUIter lru_pos;
    };

    std::unordered_map<size_t,CacheEntry> cache;        // data cache
    std::list<size_t> lru;
    mutable std::shared_mutex mutex;   // 读写锁

    public:
    /**
     * Construct a cache with a given size.
     * @param C_ the size of the cache in number of pages
     * @param fd_ the file descriptor of the data file
     * @param n the number of elements in the index
     */
    explicit LRUCache(size_t C_, int fd_, pgm::IOInterface interface=pgm::PSYNC) {
        this->fd = fd_;
        this->C = C_;
        this->cache_hits = 0;
        this->cache_misses = 0;
        this->IO_time = 0;
        this->IOs = 0;
        switch (interface){
            case pgm::PSYNC:
                this->io = std::make_unique<SyncInterface>(fd);
                break;
            case pgm::LIBAIO:
                // this->io = 
                break;
            case pgm::IO_URING:
                break;
        }
    }

    /**
     * Get segments in a page from the cache.
     * @param index the index of the page to retrieve
     *
     */
    pgm::Page& get (const size_t index) override{
        {
            std::unique_lock wlock(mutex);
            auto it = cache.find(index);
            if (it != cache.end()) {
                // Cache hit
                // update LRU
                this->cache_hits++;
                lru.erase(it->second.lru_pos);
                lru.push_front(index);
                it->second.lru_pos = lru.begin();
                return it->second.page;
            }
        }
        
        miss:
        // Evict if full
        if (cache.size() >= C) {
            std::unique_lock wlock(mutex);
            size_t old = lru.back();
            lru.pop_back();
            cache.erase(old);
        }
        // Not found, load from disk
        std::pair<pgm::Page,pgm::IOResult> res = triggerIO(index);
        auto p = res.first;
        
        // auto p = triggerIO(index);
        // Insert new page
        std::unique_lock wlock(mutex);
        lru.push_front(index);
        cache[index] = CacheEntry{std::move(p), lru.begin()};
        this->cache_misses++;
        IO_time += res.second.ns;
        IOs++;
        return cache[index].page;    
    }

    /**
     * Get segments in a page from the cache.
     * @param lo is the index of the lowest page to retrieve
     * @param hi is the index of the highest page to retrieve
     * 
     */
    std::vector<pgm::Page> get(const size_t lo, const size_t hi) override {
        std::vector<pgm::Page> res;
        size_t miss_begin = (size_t)-1;
        std::pair<std::vector<pgm::Page>, pgm::IOResult> pair;

        std::unique_lock wlock(mutex);
        for (size_t index = lo; index <= hi; index++) {
            auto it = cache.find(index);
            if (it != cache.end()) {
                // --- flush previous miss ---
                if (miss_begin != (size_t)-1) {
                    size_t miss_len = index - miss_begin;
                    pair = triggerIO(miss_begin, miss_len);
                    auto pages = pair.first;
                    // auto pages = triggerIO(miss_begin, miss_len);
                    
                    for (size_t i = 0; i < pages.size(); i++) {
                        size_t idx = miss_begin + i;
                        lru.push_front(idx);
                        cache[idx] = CacheEntry{std::move(pages[i]), lru.begin()};
                        res.push_back(cache[idx].page);
                    }
                    miss_begin = (size_t)-1;
                    IO_time += pair.second.ns;
                    IOs += miss_len;
                }
                // --- cache hit ---
                this->cache_hits++;
                lru.erase(it->second.lru_pos);
                lru.push_front(index);
                it->second.lru_pos = lru.begin();
                res.push_back(it->second.page);

            } else {
                // --- cache miss ---
                if (miss_begin == (size_t)-1) miss_begin = index;
                this->cache_misses++;
            }
            // eviction
            while (cache.size() >= C) {
                size_t old = lru.back(); lru.pop_back();
                cache.erase(old);
            }
        }

        // --- flush tail miss ---
        if (miss_begin != (size_t)-1) {
            size_t miss_len = hi - miss_begin + 1;
            pair = triggerIO(miss_begin, miss_len);
            auto pages = pair.first;
            // auto pages = triggerIO(miss_begin, miss_len);
            IO_time += pair.second.ns;
            IOs += miss_len;
            for (size_t i = 0; i < pages.size(); i++) {
                size_t idx = miss_begin + i;
                if (cache.find(idx) != cache.end()) continue;
                lru.push_front(idx);
                cache[idx] = CacheEntry{std::move(pages[i]), lru.begin()};
                res.push_back(cache[idx].page);
            }
        }
        // eviction
        while (cache.size() >= C) {
            size_t old = lru.back(); lru.pop_back();
            cache.erase(old);
        }

        return res;
    }

    void clear() override {
        std::unique_lock wlock(mutex);
        cache.clear();
        lru.clear();
    }

    size_t get_hit_count() const override { return cache_hits; }
    size_t get_miss_count() const override { return cache_misses; }
    size_t get_IO_time() const override { return IO_time; }
    size_t get_IOs() const override {return IOs; }
};