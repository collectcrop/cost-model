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
#include <list>
#include <iostream>
#include <chrono>

using timer = std::chrono::high_resolution_clock;
#define PAGE_SIZE 4096
#define MAX_FREQ 100000

class LFUCache : public ICache<size_t, Page> {
    int min_freq;

    using LFUIter = std::list<size_t>::iterator;

    struct CacheEntry {
        Page page;
        size_t freq;
        LFUIter lfu_pos;
    };

    std::unordered_map<size_t, CacheEntry> cache;
    std::unordered_map<size_t, std::list<size_t>> freq_to_keys;

    public:

    /**
     * Construct a cache with a given size.
     * @param C_ the size of the cache in number of pages
     * @param fd_ the file descriptor of the data file
     * @param n the number of elements in the index
     */
    explicit LFUCache(size_t C_, int fd_) {
        this->fd = fd_;
        this->C = C_;
        this->cache_hits = 0;
        this->cache_misses = 0;
        this->IO_time = 0;
        this->IOs = 0;
        min_freq = 0;
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
            // adapt frequency
            size_t freq = it->second.freq;
            freq_to_keys[freq].erase(it->second.lfu_pos);
            if (it->second.freq < MAX_FREQ) it->second.freq++;
            freq_to_keys[freq+1].push_back(index);
            it->second.lfu_pos = --freq_to_keys[freq+1].end();

            if (freq_to_keys[freq].empty()) {
                freq_to_keys.erase(freq);
                if (min_freq == freq) min_freq++;
            }
            return it->second.page;
        }
        this->cache_misses++;
        // Not found, load from disk
        Page p = triggerIO(index);

        if (cache.size() >= C) {
            // evict if full
            size_t old = freq_to_keys[min_freq].front();
            freq_to_keys[min_freq].pop_front();
            if (freq_to_keys[min_freq].empty()) {
                freq_to_keys.erase(min_freq);
            }
            cache.erase(old);
        }

        // Insert new page
        freq_to_keys[1].push_back(index);
        cache[index] = CacheEntry{std::move(p), 1, --freq_to_keys[1].end()};
    
        min_freq = 1;
        
        return cache[index].page;    
    }


    void clear() override {
        cache.clear();
        freq_to_keys.clear();
        min_freq = 0;
    }

    size_t get_hit_count() const override { return cache_hits; }
    size_t get_miss_count() const override { return cache_misses; }
    size_t get_IO_time() const override { return IO_time; }
    size_t get_IOs() const override {return IOs; }
};