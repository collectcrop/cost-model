// CacheInterface.hpp
#pragma once

#include <liburing.h>
#include <atomic>
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
#include <mutex>
#include <iostream>
#include <chrono>
#include "utils/include.hpp"
#include "IO/io_interface.hpp"

using timer = std::chrono::high_resolution_clock;

// struct UringMeta {
//     size_t page_index;          // file page index
//     size_t pos;                 // position in the caller's indices array
//     std::shared_ptr<char> buf;  // buffer holding the page data, deleter calls free()
//     UringMeta(size_t pi, size_t p, std::shared_ptr<char> b)
//         : page_index(pi), pos(p), buf(std::move(b)) {}
// };


// struct IOResult {
//     long long ns;    // elapsed nanoseconds
//     int64_t res;     // returned bytes or negative error
// };

struct RangeMeta {
    size_t start_index;
    size_t page_count;
    std::vector<size_t> query_positions; // indices 中对应的位置
    std::shared_ptr<char> buf;
};

// Helper: per-thread io_uring instance
// static thread_local io_uring t_ring;
// static thread_local bool t_ring_inited = false;
// static std::mutex g_ring_submit_mtx; 


// Helper to alloc page buffer and wrap in shared_ptr with free() deleter
static std::shared_ptr<char> alloc_aligned_page() {
    void* raw = nullptr;
    if (posix_memalign(&raw, pgm::PAGE_SIZE, pgm::PAGE_SIZE) != 0) {
        throw std::runtime_error("posix_memalign failed");
    }
    return std::shared_ptr<char>(reinterpret_cast<char*>(raw),
                                 [](char* p){ free(p); });
}

// static void ensure_thread_ring_init(unsigned queue_depth = 256) {
//     if (t_ring_inited) return;
//     int ret = io_uring_queue_init((unsigned)queue_depth, &t_ring, IORING_SETUP_IOPOLL);
//     if (ret < 0) {
//         throw std::runtime_error(std::string("io_uring_queue_init failed: ") + std::to_string(-ret));
//     }
//     t_ring_inited = true;
// }

// global ring and init flag
static io_uring g_ring;
static std::atomic<bool> g_ring_inited{false};
static std::mutex g_ring_init_mtx;

void ensure_global_ring_init(unsigned queue_depth = 64, int sq_thread_cpu = -1, unsigned sq_thread_idle_ms = 2000) {
    if (g_ring_inited.load(std::memory_order_acquire)) return;

    std::lock_guard<std::mutex> lg(g_ring_init_mtx);
    if (g_ring_inited.load(std::memory_order_acquire)) return;

    struct io_uring_params params;
    memset(&params, 0, sizeof(params));
    params.flags = IORING_SETUP_SQPOLL;            // enable SQPOLL
    params.sq_thread_idle = sq_thread_idle_ms;     // ms
    if (sq_thread_cpu >= 0) params.sq_thread_cpu = sq_thread_cpu;

    int ret = io_uring_queue_init_params(queue_depth, &g_ring, &params);
    if (ret < 0) {
        throw std::runtime_error(std::string("io_uring_queue_init_params failed: ") + std::to_string(-ret));
    }

    // Optionally register files or buffers here for better perf.
    // e.g. io_uring_register_files(&g_ring, files, n_files);
    // or register buffers: io_uring_register_buffers(...)

    g_ring_inited.store(true, std::memory_order_release);
}




template <typename Key, typename Value>
class ICache {
protected:
    size_t cache_misses;
    size_t cache_hits;
    size_t IO_time;
    size_t IOs;
    int fd;
    size_t C;       // Number of buffered pages
    std::unique_ptr<IOInterface> io;
public:
    virtual ~ICache() = default;
    // get cache value
    virtual Value& get(const Key index) = 0;
    virtual std::vector<Value> get(const Key lo, const Key hi) = 0;

    std::pair<pgm::Page, pgm::IOResult> triggerIO(size_t index) {
        return io->triggerIO(index);
    }

    std::pair<std::vector<pgm::Page>, pgm::IOResult> triggerIO(size_t index, size_t len) {
        return io->triggerIO(index,len);
    }

    std::pair<std::vector<pgm::Page>, pgm::IOResult> triggerIO_batch(const std::vector<size_t>& indices) {
        return io->triggerIO_batch(indices);
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
