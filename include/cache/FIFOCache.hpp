#include "CacheInterface.hpp"
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
#include <queue>
#include <iostream>
#include <chrono>

using timer = std::chrono::high_resolution_clock;
#define PAGE_SIZE 4096

class FIFOCache : public ICache<size_t, Page> {
    std::unordered_map<size_t,Page> cache;  
    std::queue<size_t> q;

    public:

    /**
     * Construct a cache with a given size.
     * @param C_ the size of the cache in number of pages
     * @param fd_ the file descriptor of the data file
     * @param n the number of elements in the index
     */
    explicit FIFOCache(size_t C_, int fd_) {
        this->fd = fd_;
        this->C = C_;
        this->cache_hits = 0;
        this->cache_misses = 0;
        this->IO_time = 0;
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
            return it->second;
        }
        this->cache_misses++;
        // Evict if full
        if (cache.size() >= C) {
            size_t old = q.front();
            q.pop();
            cache.erase(old);
        }
        // Not found, load from disk
        Page p = triggerIO(index);

        // Insert new page
        q.push(index);
        cache[index] = std::move(p);
        return cache[index];    
    }

    void clear() override {
        cache.clear();
        std::queue<size_t> empty;
        std::swap(q, empty);
    }

    size_t get_hit_count() const override { return cache_hits; }
    size_t get_miss_count() const override { return cache_misses; }
    size_t get_IO_time() const override { return IO_time; }
};