#pragma once
#include <vector>
#include <future>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>
#include <atomic>
#include <optional>
#include <algorithm>
#include <unordered_map>

#include "utils/include.hpp"
#include "utils/utils.hpp"
#include "IO/io_interface.hpp"
#include "IO/SyncInterface.hpp"
#include "IO/LibaioInterface.hpp"
#include "IO/IOuringInterface.hpp"
#include "queue/mpmc.hpp"
#include "pgm/pgm_index.hpp"

#include "cache/CacheInterface.hpp"  
// using Clock = std::chrono::steady_clock;
using Clock = std::chrono::high_resolution_clock;
namespace falcon {

struct PointResult { bool found{false}; uint64_t key{0}; };
struct RangeResult { std::vector<uint64_t> keys; };
struct FalconStats {
    uint64_t cache_hits{0}, cache_misses{0}, cache_evictions{0}, cache_puts{0};
    uint64_t physical_ios{0}, logical_ios{0}, io_bytes{0}, io_ns{0};
    uint64_t index_ns{0};  
    uint64_t cache_ns{0};  
};

enum class ReqKind { Point, Range };

struct Req {
    ReqKind kind;
    size_t page_lo, page_hi;
    uint64_t key;
    uint64_t lo_key, hi_key;
    std::promise<PointResult>  prom_point;
    std::promise<RangeResult>  prom_range;
    uint64_t ts_dispatch_ns;
};

class Queue {
    std::mutex mtx;
    std::deque<Req> q;
public:
    void push(Req&& r) { std::lock_guard<std::mutex> lg(mtx); q.emplace_back(std::move(r)); }
    std::deque<Req> drain_all(){ std::lock_guard<std::mutex> lg(mtx); std::deque<Req> o; o.swap(q); return o; }
    bool empty(){ std::lock_guard<std::mutex> lg(mtx); return q.empty(); }
};

class Worker {
public:
    Worker(int fd,
           falcon::IOInterfaceType which,
           std::unique_ptr<falcon::ICache> cache,     // 注入缓存
           size_t max_pages_per_batch,
           long max_wait_us)
    : fd_(fd), cache_(std::move(cache)),
      max_pages_(max_pages_per_batch), max_wait_us_(max_wait_us) {
        switch (which) {
            case falcon::PSYNC:    io_.reset(new SyncInterface(fd));   break;
            case falcon::LIBAIO:   io_.reset(new LibaioInterface(fd)); break;
            case falcon::IO_URING: io_.reset(new IoUringInterface(fd));break;
        }
        th_ = std::thread([this]{ this->run(); });
    }

    ~Worker() {
        stop_.store(true, std::memory_order_release);
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }

    void submit(Req&& r) { in_.push(std::move(r)); cv_.notify_one(); }

    const falcon::ICache& cache() const { return *cache_; }
    FalconStats stats() const {
        FalconStats s{};
        const auto& cs = cache_->stats();
        // s.cache_hits      = cs.hits.load(std::memory_order_relaxed);
        // s.cache_misses    = cs.misses.load(std::memory_order_relaxed);
        s.cache_hits      = logical_hits_.load(std::memory_order_relaxed);
        s.cache_misses    = logical_misses_.load(std::memory_order_relaxed);
        s.cache_evictions = cs.evictions.load(std::memory_order_relaxed);
        s.cache_puts      = cs.puts.load(std::memory_order_relaxed);
        s.physical_ios    = iostat_ios_.load(std::memory_order_relaxed);
        s.logical_ios     = iostat_logical_ios_.load(std::memory_order_relaxed);
        s.io_bytes        = iostat_bytes_.load(std::memory_order_relaxed);
        s.io_ns           = iostat_ns_.load(std::memory_order_relaxed);
        s.cache_ns        = cache_ns_.load(std::memory_order_relaxed);
        return s;
    }

private:
    int fd_;
    // std::unique_ptr<falcon::ICache> io_cache_;
    std::unique_ptr<falcon::IOInterface> io_;
    std::unique_ptr<falcon::ICache> cache_;

    size_t max_pages_;
    long   max_wait_us_;
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> iostat_ios_{0},iostat_logical_ios_{0}, iostat_bytes_{0}, iostat_ns_{0}, cache_ns_{0};
    std::atomic<uint64_t> logical_hits_{0};
    std::atomic<uint64_t> logical_misses_{0};
    std::condition_variable cv_;
    std::mutex cv_mtx_;
    // MpmcQueue<Req> in_{1024*1024};
    Queue in_;
    std::thread th_;

    void run() {
        while (!stop_.load(std::memory_order_acquire)) {
            {
                std::unique_lock<std::mutex> lk(cv_mtx_);
                cv_.wait_for(lk, std::chrono::microseconds(max_wait_us_), [&]{
                    return stop_.load(std::memory_order_acquire) || !in_.empty();
                });
            }
            auto batch = in_.drain_all();
            if (batch.empty()) continue;

            auto c0 = Clock::now();
            // 1) 汇总页需求
            std::vector<size_t> pages;
            pages.reserve(1024);
            std::unordered_map<size_t, size_t> freq;
            freq.reserve(1024);
            for (auto &r : batch) {
                for (size_t p = r.page_lo; p <= r.page_hi; ++p) {
                    pages.push_back(p);
                    ++freq[p];
                    // if (pages.size() >= max_pages_) break;
                }
            }
            if (pages.empty()) {
                for (auto &r : batch) fulfill(r, {});
                continue;
            }
            std::sort(pages.begin(), pages.end());
            pages.erase(std::unique(pages.begin(), pages.end()), pages.end());

            // 2) 先查缓存：命中页直接准备好；未命中页放 need_io
            std::unordered_map<size_t, falcon::Page> ready;
            ready.reserve(pages.size()*2);
            std::vector<size_t> need_io;
            need_io.reserve(pages.size());
            for (auto pgidx : pages) {
                size_t m = freq[pgidx];     // 这个页在本 batch 被访问了 m 次
                falcon::Page ph;
                if (cache_->get(pgidx, ph) && ph.data && ph.valid_len > 0) {
                    ready.emplace(pgidx, std::move(ph));
                    logical_hits_.fetch_add(m, std::memory_order_relaxed);
                } else {
                    need_io.push_back(pgidx);
                    // 第一条访问触发 1 次 miss
                    logical_misses_.fetch_add(1, std::memory_order_relaxed);
                    // 剩下 (m-1) 次复用刚读入的页，可以视作命中
                    if (m > 1) {
                        logical_hits_.fetch_add(m - 1, std::memory_order_relaxed);
                    }
                }
            }
            
            // 3) 对缺页进行一次性批量 I/O
            if (!need_io.empty()) {
                auto [vec, io_stat] = io_->triggerIO_batch(need_io);
                iostat_ios_.fetch_add(io_stat.physical_ios, std::memory_order_relaxed);
                iostat_logical_ios_.fetch_add(io_stat.logical_ios, std::memory_order_relaxed);
                iostat_bytes_.fetch_add(io_stat.bytes, std::memory_order_relaxed);
                iostat_ns_.fetch_add(io_stat.ns, std::memory_order_relaxed);

                // 回灌缓存 + 加入 ready
                for (size_t i = 0; i < need_io.size(); ++i) {
                    size_t pgidx = need_io[i];
                    falcon::Page page = std::move(vec[i]); // 可能为空（错误或文件尾）
                    if (page.data && page.valid_len > 0) {
                        cache_->put(pgidx, falcon::Page{ page.data, page.valid_len }); // 共享句柄
                        ready.emplace(pgidx, std::move(page));
                    } else {
                        // 留空页进入 ready；后续 fulfill 会跳过
                        ready.emplace(pgidx, falcon::Page{});
                    }
                }
            }
            auto c1 = Clock::now();
            cache_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(c1 - c0).count(), std::memory_order_relaxed);
            
            // 4) 兑现查询
            for (auto &r : batch) fulfill(r, ready);
        }
    }

    void fulfill(Req& r, const std::unordered_map<size_t, falcon::Page>& page_map) {
        if (r.kind == ReqKind::Point) {
            PointResult ret{};
            ret.key = r.key;
            for (size_t p = r.page_lo; p <= r.page_hi; ++p) {
                auto it = page_map.find(p);
                if (it == page_map.end()) continue;
                const auto& pg = it->second;
                if (!pg.data || pg.valid_len == 0) continue;
                auto* recs = reinterpret_cast<falcon::Record*>(pg.data.get());
                size_t cnt = pg.valid_len / sizeof(falcon::Record);
                if (recs[0].key > r.key || recs[cnt-1].key < r.key) continue;
                if (binary_search_record(recs, 0, cnt-1, r.key)) { ret.found = true; ret.key = r.key; break; }
            }
            r.prom_point.set_value(std::move(ret));
        } else {
            RangeResult ret{};
            for (size_t p = r.page_lo; p <= r.page_hi; ++p) {
                auto it = page_map.find(p);
                if (it == page_map.end()) continue;
                const auto& pg = it->second;
                if (!pg.data || pg.valid_len == 0) continue;

                auto* recs = reinterpret_cast<falcon::Record*>(pg.data.get());
                size_t cnt = pg.valid_len / sizeof(falcon::Record);
                if (recs[cnt-1].key < r.lo_key) continue;
                if (recs[0].key > r.hi_key) break;

                auto* lb = std::lower_bound(recs, recs+cnt, r.lo_key,
                                [](const falcon::Record& a, uint64_t k){ return a.key < k; });
                auto* ub = std::upper_bound(recs, recs+cnt, r.hi_key,
                                [](uint64_t k, const falcon::Record& a){ return k < a.key; });
                for (auto* cur = lb; cur < ub; ++cur) ret.keys.push_back(cur->key);
            }
            r.prom_range.set_value(std::move(ret));
        }
    }
};

// 对外：PGM 封装（含缓存配置）
template <typename K, size_t Eps, size_t EpsRec, typename Fp=float>
class FalconPGM {
public:
    using IndexT = pgm::PGMIndex<K,Eps,EpsRec,Fp>;

    FalconPGM(IndexT& index,
              int data_fd,
              falcon::IOInterfaceType io_kind,
              size_t memory_budget_bytes,                 
              falcon::CachePolicy cache_policy = falcon::CachePolicy::LRU,
              size_t cache_shards = 1,
              size_t max_pages_per_batch = 128,
              long max_wait_us = 50,
              size_t workers = 1)
    : idx_(index) {
        workers_.reserve(workers);
        for (size_t i = 0; i < workers; ++i) {
            auto cache = falcon::MakeShardedCache(cache_policy,
                                               memory_budget_bytes,
                                               falcon::PAGE_SIZE,
                                               cache_shards);
            workers_.emplace_back(new Worker(data_fd, io_kind,
                                             std::move(cache),
                                             max_pages_per_batch, max_wait_us));
        }
        rr_ = 0;
    }

    std::future<PointResult> point_lookup(const K& key) {
        auto t0 = Clock::now();
        auto [plo, phi] = idx_.estimate_pages_for_key(key);
        auto t1 = Clock::now();
        index_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(), std::memory_order_relaxed);
        Req r; r.kind = ReqKind::Point; r.page_lo = plo; r.page_hi = phi; r.key = (uint64_t)key;
        std::promise<PointResult> p; auto fut = p.get_future(); r.prom_point = std::move(p);
        r.ts_dispatch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        dispatch(std::move(r)); return fut;
    }

    std::future<RangeResult> range_lookup(const K& lo_key, const K& hi_key) {
        auto t0 = Clock::now();
        auto [plo, phi] = idx_.estimate_pages_for_range(lo_key, hi_key);
        auto t1 = Clock::now();
        index_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(), std::memory_order_relaxed);
        Req r; r.kind = ReqKind::Range; r.page_lo = plo; r.page_hi = phi; r.lo_key = lo_key; r.hi_key = hi_key;
        std::promise<RangeResult> p; auto fut = p.get_future(); r.prom_range = std::move(p);
        dispatch(std::move(r)); return fut;
    }

    FalconStats stats() const {
        FalconStats agg{};
        for (auto& w : workers_) {
            auto s = w->stats();
            agg.cache_hits      += s.cache_hits;
            agg.cache_misses    += s.cache_misses;
            agg.cache_evictions += s.cache_evictions;
            agg.cache_puts      += s.cache_puts;
            agg.physical_ios    += s.physical_ios;
            agg.logical_ios     += s.logical_ios;
            agg.io_bytes        += s.io_bytes;
            agg.io_ns           += s.io_ns;
            agg.cache_ns        += s.cache_ns;
            agg.index_ns        += index_ns_;
        }
        return agg;
    }

private:
    IndexT& idx_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<size_t> rr_;
    std::atomic<time_t> index_ns_ = 0;
    void dispatch(Req&& r) {
        auto i = rr_.fetch_add(1, std::memory_order_relaxed) % workers_.size();
        workers_[i]->submit(std::move(r));
    }
};


template <typename K, typename RMIIndexT>
class FalconRMI {
public:
    FalconRMI(RMIIndexT& index,
              int data_fd,
              falcon::IOInterfaceType io_kind,
              size_t memory_budget_bytes,
              falcon::CachePolicy cache_policy = falcon::CachePolicy::LRU,
              size_t cache_shards = 1,
              size_t max_pages_per_batch = 128,
              long max_wait_us = 50,
              size_t workers = 1,
              size_t n = 0)
        : idx_(index),n_(n) {
        workers_.reserve(workers);
        for (size_t i = 0; i < workers; ++i) {
            auto cache = falcon::MakeShardedCache(cache_policy,
                                               memory_budget_bytes,
                                               falcon::PAGE_SIZE,
                                               cache_shards); 
            workers_.emplace_back(
                std::make_unique<Worker>(data_fd,
                                         io_kind,
                                         std::move(cache),
                                         max_pages_per_batch,
                                         max_wait_us));
        }
        rr_.store(0, std::memory_order_relaxed);
    }

    std::future<PointResult> point_lookup(const K& key) {
        size_t err = 0; 
        auto t0 = Clock::now();
        auto pos = idx_.lookup(key,&err);
        auto t1 = Clock::now();
        index_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(), std::memory_order_relaxed);

        size_t pos_lo = (pos<=err)?0:pos-err;
        size_t pos_hi = pos+err;
        if (pos_hi>=n_) pos_hi = n_-1;

        // ====================================

        size_t plo = pos_lo / falcon::ITEM_PER_PAGE;
        size_t phi = pos_hi / falcon::ITEM_PER_PAGE;

        Req r; r.kind = ReqKind::Point; r.page_lo = plo; r.page_hi = phi; r.key = (uint64_t)key;
        std::promise<PointResult> p; auto fut = p.get_future(); r.prom_point = std::move(p);
        dispatch(std::move(r)); return fut;
    }

    std::future<RangeResult> range_lookup(const K& lo_key, const K& hi_key) {
        size_t err = 0; 
        auto t0 = Clock::now();
        auto pos1 = idx_.lookup(lo_key,&err);
        size_t pos_lo1 = (pos1<=err)?0:pos1-err;
        size_t pos_hi1 = pos1+err;
        if (pos_hi1>=n_) pos_hi1 = n_-1;
        auto pos2 = idx_.lookup(hi_key,&err);
        size_t pos_lo2 = (pos2<=err)?0:pos2-err;
        size_t pos_hi2 = pos2+err;
        if (pos_hi2>=n_) pos_hi2 = n_-1;

        auto t1 = Clock::now();
        index_ns_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(), std::memory_order_relaxed);
        // ====================================
        size_t pos_lo = std::min(pos_lo1, pos_lo2);
        size_t pos_hi = std::max(pos_hi1, pos_hi2);

        size_t plo = pos_lo / falcon::ITEM_PER_PAGE;
        size_t phi = pos_hi / falcon::ITEM_PER_PAGE;

        Req r; r.kind = ReqKind::Range; r.page_lo = plo; r.page_hi = phi; r.lo_key = lo_key; r.hi_key = hi_key;
        std::promise<RangeResult> p; auto fut = p.get_future(); r.prom_range = std::move(p);
        dispatch(std::move(r)); return fut;
    }

    FalconStats stats() const {
        FalconStats agg{};
        for (auto& w : workers_) {
            auto s = w->stats();
            agg.cache_hits      += s.cache_hits;
            agg.cache_misses    += s.cache_misses;
            agg.cache_evictions += s.cache_evictions;
            agg.cache_puts      += s.cache_puts;
            agg.physical_ios    += s.physical_ios;
            agg.logical_ios     += s.logical_ios;
            agg.io_bytes        += s.io_bytes;
            agg.io_ns           += s.io_ns;
            agg.cache_ns        += s.cache_ns;
            agg.index_ns        += index_ns_;
        }
        return agg;
    }


private:
    RMIIndexT& idx_;
    size_t n_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<size_t> rr_;
    std::atomic<time_t> index_ns_ = 0;
    void dispatch(Req&& r) {
        auto i = rr_.fetch_add(1, std::memory_order_relaxed) % workers_.size();
        workers_[i]->submit(std::move(r));
    }
};

} // namespace falcon