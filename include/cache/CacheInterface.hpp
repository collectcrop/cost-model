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
    
    // std::pair<Page, IOResult> / Page
    // std::pair<Page, IOResult> triggerIO(size_t index){          
    //     ensure_thread_ring_init();
    //     Page p;
    //     void* raw_ptr = nullptr;
    //     if (posix_memalign(&raw_ptr, pgm::PAGE_SIZE, pgm::PAGE_SIZE) != 0) {
    //         throw std::runtime_error("posix_memalign failed");
    //     }
    //     p.data.reset(reinterpret_cast<char*>(raw_ptr));

    //     off_t offset = index * pgm::PAGE_SIZE;
    //     auto t0 = timer::now();
    //     // ssize_t bytes = pread(fd, p.data.get(), pgm::PAGE_SIZE, offset);

    //     // get sqe
    //     struct io_uring_sqe* sqe = io_uring_get_sqe(&t_ring);
    //     if (!sqe) {
    //         free(raw_ptr);
    //         throw std::runtime_error("io_uring_get_sqe returned null");
    //     }
    //     // prepare read
    //     io_uring_prep_read(sqe, fd, p.data.get(), pgm::PAGE_SIZE, offset);

    //     // tag user_data to identify request if needed (here use pointer to buffer)
    //     io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(p.data.get()));
    //     int ret = io_uring_submit(&t_ring);
    //     if (ret < 0) {
    //         free(raw_ptr);
    //         throw std::runtime_error(std::string("io_uring_submit failed: ") + std::to_string(-ret));
    //     }
    //     // wait for completion
    //     struct io_uring_cqe* cqe = nullptr;
    //     ret = io_uring_wait_cqe(&t_ring, &cqe);

    //     auto t1 = timer::now();
    //     if (ret < 0) {
    //         io_uring_cqe_seen(&t_ring, cqe);
    //         free(raw_ptr);
    //         throw std::runtime_error(std::string("io_uring_wait_cqe failed: ") + std::to_string(-ret));
    //     }
    //     long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    //     // IO_time += ns;
    //     // IOs ++;
    //     // std::cout << "Reading page " << index << " took " << query_ns << " ns" << std::endl;

    //     int64_t result = cqe->res;
    //     io_uring_cqe_seen(&t_ring, cqe);

    //     if (result < 0) {
    //         free(raw_ptr);
    //         throw std::runtime_error(std::string("io_uring read error: ") + std::to_string((int)-result));
    //     }

    //     p.valid_len = static_cast<size_t>(result);
    //     // if (bytes < 0){
    //     //     std::cout << fd << std::endl; 
    //     //     throw std::runtime_error("Failed to read data from disk at offset " + std::to_string(offset));
    //     // }
    //     // p.valid_len = bytes;
    //     // return p;
    //     IOResult ior{ ns, result };
    //     return { std::move(p), ior };
    // }
    // //std::pair<std::vector<Page>, IOResult> / std::vector<Page>
    // std::pair<std::vector<Page>, IOResult> triggerIO(size_t index, size_t page_num){
    //     ensure_thread_ring_init();
    //     size_t agg_size = pgm::PAGE_SIZE * page_num;

    //     std::vector<Page> res;
    //     // first read to a Aggregated page
    //     void* raw_ptr = nullptr;
    //     if (posix_memalign(&raw_ptr, pgm::PAGE_SIZE, pgm::PAGE_SIZE * page_num) != 0) {
    //         throw std::runtime_error("posix_memalign failed");
    //     }
    //     std::shared_ptr<char[]> agg_data(reinterpret_cast<char*>(raw_ptr));

    //     off_t offset = index * pgm::PAGE_SIZE;
    //     auto t0 = timer::now();
    //     // ssize_t bytes = pread(fd, agg_data.get(), pgm::PAGE_SIZE * page_num, offset);
    //     struct io_uring_sqe* sqe = io_uring_get_sqe(&t_ring);
    //     if (!sqe) throw std::runtime_error("io_uring_get_sqe returned null (agg)");

    //     io_uring_prep_read(sqe, fd, agg_data.get(), agg_size, offset);
    //     io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(agg_data.get()));
    //     int ret = io_uring_submit(&t_ring);
    //     if (ret < 0) throw std::runtime_error(std::string("io_uring_submit failed: ") + std::to_string(-ret));

    //     struct io_uring_cqe* cqe = nullptr;
    //     ret = io_uring_wait_cqe(&t_ring, &cqe);

    //     auto t1 = timer::now();
    //     long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    //     // IO_time += ns;
    //     // IOs += page_num;
    //     if (ret < 0) {
    //         io_uring_cqe_seen(&t_ring, cqe);
    //         throw std::runtime_error(std::string("io_uring_wait_cqe failed: ") + std::to_string(-ret));
    //     }
    //     // if (bytes < 0){
    //     //     std::cout << fd << std::endl; 
    //     //     throw std::runtime_error("Failed to read data from disk at offset " + std::to_string(offset));
    //     // }
    //     int64_t bytes = cqe->res;
    //     io_uring_cqe_seen(&t_ring, cqe);
    //     if (bytes < 0) {
    //         throw std::runtime_error(std::string("io_uring agg read error: ") + std::to_string((int)-bytes));
    //     }

    //     // then separate each page from Aggregated page
    //     size_t pages_read = (bytes + pgm::PAGE_SIZE - 1) / pgm::PAGE_SIZE;
    //     for (size_t i = 0; i < pages_read; i++) {
    //         Page page;
    //         void* page_ptr = nullptr;
    //         if (posix_memalign(&page_ptr, pgm::PAGE_SIZE, pgm::PAGE_SIZE) != 0) {
    //             throw std::runtime_error("posix_memalign failed for sub-page");
    //         }
    //         page.data.reset(reinterpret_cast<char*>(page_ptr));

    //         size_t copy_size = std::min(static_cast<size_t>(bytes - i * pgm::PAGE_SIZE), (size_t)pgm::PAGE_SIZE);
    //         memcpy(page.data.get(), agg_data.get() + i * pgm::PAGE_SIZE, copy_size);

    //         page.valid_len = copy_size;
    //         res.push_back(std::move(page));
    //     }

    //     // return std::move(res);
    //     IOResult ior{ ns, bytes };
    //     return { std::move(res), ior };
    // }


    /**
     * Batch triggerIO: submit reads for all indices (one page per index).
     * Returns pages in the same order as the input indices vector.
     *
     * NOTE: io_depth is respected implicitly (we will submit SQEs in loop and
     * call io_uring_submit() whenever get_sqe() returns NULL to flush).
     */
    // std::pair<std::vector<pgm::Page>, pgm::IOResult>
    // triggerIO_batch(const std::vector<size_t>& indices) {
    //     IOResult stats;
    //     stats.logical_ios = indices.size();

    //     if (indices.empty()) return {{}, stats};
    //     ensure_global_ring_init();

    //     size_t n = indices.size();
    //     std::vector<pgm::Page> result(n);

    //     // ---- Step 1. 收集并排序请求 ----
    //     struct Req { size_t page_index; size_t query_pos; };
    //     std::vector<Req> reqs;
    //     reqs.reserve(n);
    //     for (size_t i = 0; i < n; i++) {
    //         reqs.push_back({indices[i], i});
    //     }
    //     std::sort(reqs.begin(), reqs.end(),
    //             [](auto& a, auto& b){ return a.page_index < b.page_index; });

    //     // ---- Step 2. 去重 + 合并相邻 ----
    //     std::vector<RangeMeta*> ranges;
    //     for (size_t i = 0; i < reqs.size(); i++) {
    //         if (ranges.empty()) {
    //             auto r = new RangeMeta();
    //             r->start_index = reqs[i].page_index;
    //             r->page_count = 1;
    //             r->query_positions.push_back(reqs[i].query_pos);
    //             ranges.push_back(r);
    //         } else {
    //             stats.physical_ios++;
    //             auto* last = ranges.back();
    //             size_t expected_next = last->start_index + last->page_count;
    //             if (reqs[i].page_index == expected_next) {
    //                 // 连续 -> 合并
    //                 last->page_count++;
    //                 last->query_positions.push_back(reqs[i].query_pos);
    //             } else if (reqs[i].page_index < expected_next) {
    //                 // 重复的 index
    //                 last->query_positions.push_back(reqs[i].query_pos);
    //                 stats.physical_ios--;
    //             } else {
    //                 // 断开 -> 新区间
    //                 auto r = new RangeMeta();
    //                 r->start_index = reqs[i].page_index;
    //                 r->page_count = 1;
    //                 r->query_positions.push_back(reqs[i].query_pos);
    //                 ranges.push_back(r);
    //             }
    //         }
    //     }

    //     // ---- Step 3. 提交所有 IO ----
    //     {
    //         std::unique_lock<std::mutex> lk(g_ring_submit_mtx);

    //         for (auto* r : ranges) {
    //             size_t bytes = r->page_count * pgm::PAGE_SIZE;

    //             // 分配 buffer
    //             void* raw = nullptr;
    //             if (posix_memalign(&raw, pgm::PAGE_SIZE, bytes) != 0) {
    //                 throw std::runtime_error("posix_memalign failed");
    //             }
    //             r->buf = std::shared_ptr<char>((char*)raw, [](char* p){ free(p); });

    //             // 准备 SQE
    //             struct io_uring_sqe* sqe = io_uring_get_sqe(&g_ring);
    //             if (!sqe) {
    //                 // flush
    //                 int ret = io_uring_submit(&g_ring);
    //                 if (ret < 0) throw std::runtime_error("io_uring_submit failed");
    //                 sqe = io_uring_get_sqe(&g_ring);
    //                 if (!sqe) throw std::runtime_error("io_uring_get_sqe null");
    //             }
    //             off_t offset = (off_t)r->start_index * (off_t)pgm::PAGE_SIZE;
    //             io_uring_prep_read(sqe, fd, r->buf.get(), bytes, offset);
    //             io_uring_sqe_set_data(sqe, r);
    //         }

    //         int ret = io_uring_submit(&g_ring);
    //         if (ret < 0) throw std::runtime_error("io_uring_submit failed");
    //     }

    //     // ---- Step 4. 等待完成并分发 ----
    //     size_t completed = 0;
    //     auto t0 = timer::now();
    //     while (completed < ranges.size()) {
    //         struct io_uring_cqe* cqe = nullptr;
    //         int ret = io_uring_wait_cqe(&g_ring, &cqe);
    //         if (ret == -EINTR) continue;
    //         if (ret < 0) throw std::runtime_error("io_uring_wait_cqe failed");

    //         auto* r = reinterpret_cast<RangeMeta*>(io_uring_cqe_get_data(cqe));
    //         ssize_t res = cqe->res;
    //         io_uring_cqe_seen(&g_ring, cqe);

    //         if (res < 0) {
    //             // 整个区间 IO 出错 -> 返回空页
    //             for (auto pos : r->query_positions) {
    //                 result[pos] = pgm::Page{};
    //             }
    //         } else {
    //             stats.bytes += res;
    //             size_t got_pages = (res + pgm::PAGE_SIZE - 1) / pgm::PAGE_SIZE;
    //             for (size_t i = 0; i < r->query_positions.size(); i++) {
    //                 size_t pos = r->query_positions[i];
    //                 size_t rel_page = indices[pos] - r->start_index;
    //                 if (rel_page < got_pages) {
    //                     pgm::Page p;
    //                     void* sub_raw = nullptr;
    //                     if (posix_memalign(&sub_raw, pgm::PAGE_SIZE, pgm::PAGE_SIZE) != 0) {
    //                         throw std::runtime_error("posix_memalign failed (sub)");
    //                     }
    //                     std::shared_ptr<char[]> sub_buf((char*)sub_raw, [](char* p){ free(p); });
    //                     memcpy(sub_buf.get(),
    //                         r->buf.get() + rel_page * pgm::PAGE_SIZE,
    //                         pgm::PAGE_SIZE);
    //                     p.data = std::move(sub_buf);
    //                     p.valid_len = pgm::PAGE_SIZE;
    //                     result[pos] = std::move(p);
    //                 } else {
    //                     result[pos] = pgm::Page{}; // 没读到
    //                 }
    //             }
    //         }
    //         delete r;
    //         completed++;
    //     }
    //     auto t1 = timer::now();
    //     stats.ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    //     return { std::move(result), stats };
    // }

    
    // // convenience single-index wrapper (keeps compatibility)
    // std::pair<pgm::Page, pgm::IOResult> triggerIO(size_t index) {
    //     auto batch = std::vector<size_t>{ index };
    //     auto pr = triggerIO_batch(batch);
    //     if (pr.first.empty()) throw std::runtime_error("triggerIO_single: no page returned");
    //     return { std::move(pr.first[0]), pr.second };
    // }

    // clear cache
    virtual void clear() = 0;

    // statistics
    virtual size_t get_hit_count() const = 0;
    virtual size_t get_miss_count() const = 0;
    virtual size_t get_IO_time() const = 0;
    virtual size_t get_IOs() const = 0;
    size_t get_C() const { return C; }
};
