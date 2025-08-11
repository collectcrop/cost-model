// CacheInterface.hpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstring>
#include <type_traits>
#include <utility>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>  
#include <unordered_map>
#include <memory>
#include <iostream>
#include <chrono>

struct Page{
    std::unique_ptr<char[]> data;
    size_t valid_len=0;
};
using timer = std::chrono::high_resolution_clock;
#define PAGE_SIZE 4096

template <typename Key, typename Value>
class ICache {
protected:
    size_t cache_misses;
    size_t cache_hits;
    size_t IO_time;
    size_t IOs;
    int fd;
    size_t C;       // Number of buffered pages

public:
    virtual ~ICache() = default;
    // get cache value
    virtual Value& get(const Key key) = 0;
    Page triggerIO(size_t index){
        Page p;
        p.data.reset();
        void* raw_ptr = nullptr;
        if (posix_memalign(&raw_ptr, PAGE_SIZE, PAGE_SIZE) != 0) {
            throw std::runtime_error("posix_memalign failed");
        }
        p.data.reset(reinterpret_cast<char*>(raw_ptr));

        off_t offset = index * PAGE_SIZE;
        auto t0 = timer::now();
        ssize_t bytes = pread(fd, p.data.get(), PAGE_SIZE, offset);
        auto t1 = timer::now();
        auto query_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        // std::cout << "Reading page " << index << " took " << query_ns << " ns" << std::endl;
        IO_time += query_ns;
        IOs++;
        if (bytes < 0){
            std::cout << fd << std::endl; 
            throw std::runtime_error("Failed to read data from disk at offset " + std::to_string(offset));
        }
        p.valid_len = bytes;
        return p;
    }

    // clear cache
    virtual void clear() = 0;

    // statistics
    virtual size_t get_hit_count() const = 0;
    virtual size_t get_miss_count() const = 0;
    virtual size_t get_IO_time() const = 0;
    virtual size_t get_IOs() const = 0;
    size_t get_C() const { return C; }
};
