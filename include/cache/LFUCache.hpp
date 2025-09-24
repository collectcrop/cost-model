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
#include <shared_mutex>
#include <mutex>

using timer = std::chrono::high_resolution_clock;
#define MAX_FREQ 100000

class LFUCache : public ICache<size_t, pgm::Page> {
    int min_freq;

    using LFUIter = std::list<size_t>::iterator;

    struct CacheEntry {
        pgm::Page page;
        size_t freq;
        LFUIter lfu_pos;
    };

    std::unordered_map<size_t, CacheEntry> cache;
    std::unordered_map<size_t, std::list<size_t>> freq_to_keys;
    mutable std::shared_mutex mutex; 
    public:

    /**
     * Construct a cache with a given size.
     * @param C_ the size of the cache in number of pages
     * @param fd_ the file descriptor of the data file
     * @param n the number of elements in the index
     */
    explicit LFUCache(size_t C_, int fd_, pgm::IOInterface interface=pgm::PSYNC) {
        this->fd = fd_;
        this->C = C_;
        this->cache_hits = 0;
        this->cache_misses = 0;
        this->IO_time = 0;
        this->IOs = 0;
        min_freq = 0;
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
            std::shared_lock rlock(mutex);
            auto it = cache.find(index);
            if (it != cache.end()) {
                // Cache hit
                rlock.unlock();
                std::unique_lock wlock(mutex);
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
        }

        std::unique_lock wlock(mutex);
        this->cache_misses++;
        // Not found, load from disk
        std::pair<pgm::Page,pgm::IOResult> res = triggerIO(index);
        auto p = res.first;
        // auto p = triggerIO(index);

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
        std::unique_lock wlock(mutex);
        std::pair<std::vector<pgm::Page>, pgm::IOResult> pair;

        for (size_t index = lo; index <= hi; index++) {
            auto it = cache.find(index);
            if (it != cache.end()) {
                // --- flush previous miss ---
                if (miss_begin != (size_t)-1) {
                    size_t miss_len = index - miss_begin;
                    pair = triggerIO(miss_begin, miss_len);
                    auto pages = pair.first;
                    // auto pages = triggerIO(miss_begin, miss_len);
                    IO_time += pair.second.ns;
                    IOs += miss_len;
                    for (size_t i = 0; i < pages.size(); i++) {
                        size_t idx = miss_begin + i;
                        freq_to_keys[1].push_back(idx);
                        cache[idx] = CacheEntry{std::move(pages[i]), 1, --freq_to_keys[1].end()};
                        res.push_back(cache[idx].page);
                    }
                    miss_begin = (size_t)-1;
                    min_freq = 1;
                }

                // --- cache hit ---
                this->cache_hits++;
                size_t freq = it->second.freq;
                freq_to_keys[freq].erase(it->second.lfu_pos);
                if (it->second.freq < MAX_FREQ) it->second.freq++;
                freq_to_keys[it->second.freq].push_back(index);
                it->second.lfu_pos = --freq_to_keys[it->second.freq].end();

                if (freq_to_keys[freq].empty()) {
                    freq_to_keys.erase(freq);
                    if (min_freq == freq) min_freq++;
                }

                res.push_back(it->second.page);

            } else {
                // --- cache miss ---
                if (miss_begin == (size_t)-1) miss_begin = index;
                this->cache_misses++;
            }
            
            while (cache.size() >= C) {
                // evict LFU
                size_t old = freq_to_keys[min_freq].front();
                freq_to_keys[min_freq].pop_front();
                if (freq_to_keys[min_freq].empty()) {
                    freq_to_keys.erase(min_freq);
                }
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
                freq_to_keys[1].push_back(idx);
                cache[idx] = CacheEntry{std::move(pages[i]), 1, --freq_to_keys[1].end()};
                res.push_back(cache[idx].page);
            }
            min_freq = 1;
        }
        
        while (cache.size() >= C) {
            // evict LFU
            size_t old = freq_to_keys[min_freq].front();
            freq_to_keys[min_freq].pop_front();
            if (freq_to_keys[min_freq].empty()) {
                freq_to_keys.erase(min_freq);
            }
            cache.erase(old);
        }
        return res;
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