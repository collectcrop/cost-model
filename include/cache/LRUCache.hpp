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

using timer = std::chrono::high_resolution_clock;
#define PAGE_SIZE 4096


class LRUCache : public ICache<size_t, Page> {
    using LRUIter = std::list<size_t>::iterator;

    struct CacheEntry {
        Page page;
        LRUIter lru_pos;
    };

    std::unordered_map<size_t,CacheEntry> cache;        // data cache
    std::list<size_t> lru;

    public:
    /**
     * Construct a cache with a given size.
     * @param C_ the size of the cache in number of pages
     * @param fd_ the file descriptor of the data file
     * @param n the number of elements in the index
     */
    explicit LRUCache(size_t C_, int fd_) {
        this->fd = fd_;
        this->C = C_;
        this->cache_hits = 0;
        this->cache_misses = 0;
        this->IO_time = 0;
        this->IOs = 0;
    }

    /**
     * Get segments in a page from the cache.
     * @param index the index of the page to retrieve
     *
     */
    Page& get (const size_t index) override{
        auto it = cache.find(index);
        if (it != cache.end()) {
            // Cache hit
            this->cache_hits++;
            lru.erase(it->second.lru_pos);
            lru.push_front(index);
            it->second.lru_pos = lru.begin();
            return it->second.page;
        }
        this->cache_misses++;
        // Evict if full
        if (cache.size() >= C) {
            size_t old = lru.back();
            lru.pop_back();
            cache.erase(old);
        }


        // Not found, load from disk
        Page p = triggerIO(index);

        // Insert new page
        lru.push_front(index);
        cache[index] = CacheEntry{std::move(p), lru.begin()};
        return cache[index].page;    
    }

    void clear() override {
        cache.clear();
        lru.clear();
    }

    size_t get_hit_count() const override { return cache_hits; }
    size_t get_miss_count() const override { return cache_misses; }
    size_t get_IO_time() const override { return IO_time; }
    size_t get_IOs() const override {return IOs; }
};