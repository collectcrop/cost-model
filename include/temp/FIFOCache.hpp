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
#include <shared_mutex>
#include <mutex>
#include "IO/SyncInterface.hpp"
#include "IO/LibaioInterface.hpp"
#include "IO/IOuringInterface.hpp"

using timer = std::chrono::high_resolution_clock;

class FIFOCache : public ICache<size_t, pgm::Page> {
    std::unordered_map<size_t,pgm::Page> cache;  
    std::queue<size_t> q;
    mutable std::shared_mutex mutex; 
    public:

    /**
     * Construct a cache with a given size.
     * @param C_ the size of the cache in number of pages
     * @param fd_ the file descriptor of the data file
     * @param n the number of elements in the index
     */
    explicit FIFOCache(size_t C_, int fd_, pgm::IOInterface interface=pgm::PSYNC) {
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
                this->io = std::make_unique<LibaioInterface>(fd);
                break;
            case pgm::IO_URING:
                this->io = std::make_unique<IoUringInterface>(fd);
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
                this->cache_hits++;
                return it->second;
            }
        }
        std::unique_lock wlock(mutex);
        this->cache_misses++;
        // Evict if full
        if (cache.size() >= C) {
            size_t old = q.front();
            q.pop();
            cache.erase(old);
        }
        // Not found, load from disk
        std::pair<pgm::Page,pgm::IOResult> res = triggerIO(index);
        pgm::Page p = res.first;
        // Page p = triggerIO(index);
        // Insert new page
        q.push(index);
        cache[index] = std::move(p);
        IO_time += res.second.ns;
        IOs++;
        return cache[index];    
    }

    /**
     * Get segments in a page from the cache.
     * @param lo is the index of the lowest page to retrieve
     * @param hi is the index of the highest page to retrieve
     * 
     */
    std::vector<pgm::Page> get (const size_t lo, const size_t hi) override{
        std::vector<pgm::Page> res;
        size_t miss_begin = -1;
        std::unique_lock wlock(mutex);
        std::pair<std::vector<pgm::Page>, pgm::IOResult> pair;

        for (size_t index=lo;index<=hi;index++){
            auto it = cache.find(index);
            if (it != cache.end()) {
                // trigger previous IO
                if (miss_begin != (size_t)-1){
                    size_t miss_len = index - miss_begin;
                    pair = triggerIO(miss_begin, miss_len);
                    auto pages = pair.first;
                    IO_time += pair.second.ns;
                    IOs += miss_len;
                    // auto pages = triggerIO(miss_begin, miss_len);
                    for (int i=0;i<pages.size();i++){
                        size_t idx = miss_begin + i;
                        // Insert new page
                        q.push(idx);
                        cache[idx] = std::move(pages[i]);
                        res.push_back(cache[idx]);
                    }
                    miss_begin = -1;
                }
                // Cache hit
                this->cache_hits++;
                res.push_back(it->second);
            }else{
                // record miss
                if (miss_begin == (size_t)-1) miss_begin = index;
                this->cache_misses++;
            }
            // Evict if full
            while (cache.size() >= C) {
                size_t old = q.front();
                q.pop();
                cache.erase(old);
            }
        }
        // flush tail miss
        if (miss_begin != (size_t)-1) {
            size_t miss_len = hi - miss_begin + 1;
            pair = triggerIO(miss_begin, miss_len);
            auto pages = pair.first;
            IO_time += pair.second.ns;
            IOs += miss_len;
            // auto pages = triggerIO(miss_begin, miss_len);
            for (size_t i = 0; i < pages.size(); i++) {
                size_t idx = miss_begin + i;
                
                q.push(idx);
                cache[idx] = std::move(pages[i]);
                res.push_back(cache[idx]);
            }
        }
        // Evict if full
        while (cache.size() >= C) {
            size_t old = q.front();
            q.pop();
            cache.erase(old);
        }
        
        return res;    
    }

    void clear() override {
        std::unique_lock wlock(mutex);
        cache.clear();
        std::queue<size_t> empty;
        std::swap(q, empty);
    }

    size_t get_hit_count() const override { return cache_hits; }
    size_t get_miss_count() const override { return cache_misses; }
    size_t get_IO_time() const override { return IO_time; }
    size_t get_IOs() const override {return IOs; }
};