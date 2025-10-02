#pragma once
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <future>
#include <thread>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using timer = std::chrono::high_resolution_clock;
using ns_t = long long;

constexpr size_t PAGE_SIZE = 4096;

struct Page {
    std::shared_ptr<char> data; // use shared_ptr with custom deleter or shared_ptr<char[]> if prefer
    size_t valid_len = 0;
    Page() = default;
};

// result returned by reading
struct IOResult {
    ns_t ns;
    int64_t bytes;
};

inline std::shared_ptr<char> alloc_page_buffer() {
    void* p = nullptr;
    if (posix_memalign(&p, PAGE_SIZE, PAGE_SIZE) != 0) throw std::runtime_error("posix_memalign failed");
    // use shared_ptr with custom deleter to free via free()
    return std::shared_ptr<char>(reinterpret_cast<char*>(p), [](char* q){ free(q); });
}

// Request object: single page request promise/future
struct Request {
    size_t index;
    std::promise<Page> prom;
    std::shared_future<Page> fut;
    // optional: timestamp for debugging
    Request(size_t idx): index(idx) {
        fut = prom.get_future().share();
    }
};

class CachePool {
public:
    // fd: data file descriptor (opened with O_DIRECT if desired)
    // cache_capacity: max number of pages to hold
    // n_workers: number of background IO worker threads
    // batch_size: how many distinct pages a worker will try to aggregate per wake
    CachePool(size_t cache_capacity, int fd, int n_workers = 4, size_t batch_size = 64)
        : fd(fd), C(cache_capacity), stop_(false), workers(n_workers), batch_size_(batch_size)
    {
        cache_hits = 0;
        cache_misses = 0;
        IO_time = 0;
        IOs = 0;
        for (int i = 0; i < (int)workers.size(); ++i) {
            workers[i] = std::thread(&CachePool::worker_loop, this, i);
        }
    }

    ~CachePool() {
        {
            std::unique_lock<std::mutex> lk(queue_mtx_);
            stop_ = true;
            queue_cv_.notify_all();
        }
        for (auto &t : workers) if (t.joinable()) t.join();
    }

    // get cache value
    virtual Page& get(const size_t index) = 0;
    virtual std::vector<Page> get(const size_t lo, const size_t hi) = 0;

    // Expose some stats (optional)
    // void stats_print() {
    //     std::shared_lock<std::shared_mutex> rlk(cache_mtx_);
    //     std::cout << "cache size=" << cache.size() << " inflight=" << inflight_.size()
    //               << " queue_len=" << req_queue_.size() << std::endl;
    // }

    // clear cache
    virtual void clear() = 0;

    // statistics
    virtual size_t get_hit_count() const = 0;
    virtual size_t get_miss_count() const = 0;
    virtual size_t get_IO_time() const = 0;
    virtual size_t get_IOs() const = 0;
    size_t get_C() const { return C; }

protected:
    int fd;
    size_t C;
    size_t cache_misses;
    size_t cache_hits;
    size_t IO_time;
    size_t IOs;
    // cache storage
    std::unordered_map<size_t, Page> cache;
    mutable std::shared_mutex cache_mtx_;

    // inflight requests to avoid duplicated IO
    std::unordered_map<size_t, std::shared_ptr<Request>> inflight_;
    std::mutex inflight_mtx_;

    // request queue for workers
    std::deque<std::shared_ptr<Request>> req_queue_;
    std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    bool stop_;

    // worker threads
    std::vector<std::thread> workers;
    size_t batch_size_;

    // eviction policy: override this method in derived class if needed
    // virtual void evict_one() = 0;

    // worker loop
    void worker_loop(int wid) {
        while (true) {
            std::vector<std::shared_ptr<Request>> batch;
            batch.reserve(batch_size_);
            {
                std::unique_lock<std::mutex> qlk(queue_mtx_);
                queue_cv_.wait(qlk, [&]{ return stop_ || !req_queue_.empty(); });
                if (stop_ && req_queue_.empty()) return;
                // pop up to batch_size_ unique indices
                std::unordered_set<size_t> seen;
                while (!req_queue_.empty() && batch.size() < batch_size_) {
                    auto r = req_queue_.front();
                    req_queue_.pop_front();
                    if (seen.insert(r->index).second) {
                        batch.push_back(r);
                    } // duplicate index in queue => ignore duplicate occurrence
                }
            }
            if (batch.empty()) continue;

            // sort indices and group adjacent ones
            std::sort(batch.begin(), batch.end(), [](auto &a, auto &b){ return a->index < b->index; });

            // create list of contiguous ranges from batch
            size_t i = 0;
            while (i < batch.size()) {
                size_t start_idx = batch[i]->index;
                size_t j = i + 1;
                while (j < batch.size() && batch[j]->index == batch[j-1]->index + 1) ++j;
                size_t end_idx = batch[j-1]->index;
                size_t num_pages = end_idx - start_idx + 1;

                // perform aggregated read for [start_idx, end_idx]
                try {
                    auto [pages, ior] = triggerIO(start_idx, num_pages);
                    // pages should have num_pages entries (or fewer if EOF)
                    // for each page, set promise
                    for (size_t k = 0; k < pages.size(); ++k) {
                        size_t idx = start_idx + k;
                        // set value for matching request(s); some requests may not be present in current batch (rare)
                        // we must find inflight request for idx
                        std::shared_ptr<Request> rptr;
                        {
                            std::unique_lock<std::mutex> lk(inflight_mtx_);
                            auto it = inflight_.find(idx);
                            if (it != inflight_.end()) rptr = it->second;
                        }
                        if (rptr) {
                            rptr->prom.set_value(pages[k]);
                        } else {
                            // no one waiting currently, but still we might want to put into cache directly
                            std::unique_lock<std::shared_mutex> wlk(cache_mtx_);
                            if (cache.size() >= C) {
                                auto it = cache.begin();
                                cache.erase(it);
                            }
                            cache[idx] = pages[k];
                        }
                    }
                } catch (const std::exception &e) {
                    // on IO error, set exception for all involved requests
                    for (size_t k = 0; k < num_pages; ++k) {
                        size_t idx = start_idx + k;
                        std::shared_ptr<Request> rptr;
                        {
                            std::unique_lock<std::mutex> lk(inflight_mtx_);
                            auto it = inflight_.find(idx);
                            if (it != inflight_.end()) rptr = it->second;
                        }
                        if (rptr) {
                            try { rptr->prom.set_exception(std::current_exception()); }
                            catch(...) {}
                        }
                    }
                }

                i = j;
            } // end handle sorted batch
        } // end worker loop
    }

    // helper: group adjacent indices into ranges
    std::vector<std::pair<size_t,size_t>> group_adjacent(const std::vector<size_t>& misses) {
        if (misses.empty()) return {};
        std::vector<std::pair<size_t,size_t>> out;
        size_t s = misses[0], e = misses[0];
        for (size_t i = 1; i < misses.size(); ++i) {
            if (misses[i] == e + 1) e = misses[i];
            else { out.emplace_back(s,e); s = e = misses[i]; }
        }
        out.emplace_back(s,e);
        return out;
    }

    // psync implementation
    std::pair<std::vector<Page>, IOResult> triggerIO(size_t start_index, size_t num_pages) {
        size_t agg_size = num_pages * PAGE_SIZE;
        void* raw = nullptr;
        if (posix_memalign(&raw, PAGE_SIZE, agg_size) != 0) throw std::runtime_error("posix_memalign agg failed");
        std::shared_ptr<char> aggbuf(reinterpret_cast<char*>(raw), [](char* p){ free(p); });

        off_t offset = static_cast<off_t>(start_index) * static_cast<off_t>(PAGE_SIZE);
        auto t0 = timer::now();
        ssize_t br;
        // retry on EINTR
        while (true) {
            br = pread(fd, aggbuf.get(), agg_size, offset);
            if (br >= 0) break;
            if (errno == EINTR) continue;
            free(raw); // aggbuf's deleter will free; but raw is also to be freed if posix_memalign succeeded
            throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
        }
        auto t1 = timer::now();
        ns_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        size_t pages_read = static_cast<size_t>((br + PAGE_SIZE - 1) / PAGE_SIZE);
        std::vector<Page> pages;
        pages.reserve(pages_read);
        for (size_t i = 0; i < pages_read; ++i) {
            Page p;
            p.data = alloc_page_buffer();
            size_t copy_size = std::min(static_cast<size_t>(br - i * PAGE_SIZE), PAGE_SIZE);
            memcpy(p.data.get(), aggbuf.get() + i * PAGE_SIZE, copy_size);
            p.valid_len = copy_size;
            pages.push_back(std::move(p));
        }
        return {std::move(pages), IOResult{ns, br}};
    }

    // libaio implementation

    // io_uring implementation
};
