// queue/mpmc.hpp
#pragma once
#include <atomic>
#include <cstddef>
#include <deque>
#include <optional>
#include <thread>
#include <utility>
#include <vector>
#include <cassert>

template <class T>
class MpmcQueue {
        struct Cell {
        std::atomic<size_t> seq;
        std::optional<T>    data;

        Cell() noexcept : seq(0), data(std::nullopt) {}

        // 禁止拷贝
        Cell(const Cell&) = delete;
        Cell& operator=(const Cell&) = delete;

        // 允许移动：用 load/store 迁移 atomic 的值
        Cell(Cell&& o) noexcept
            : seq(o.seq.load(std::memory_order_relaxed)),
              data(std::move(o.data)) {}

        Cell& operator=(Cell&& o) noexcept {
            if (this != &o) {
                seq.store(o.seq.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
                data = std::move(o.data);
            }
            return *this;
        }
    };

    static size_t round_up_pow2(size_t x) {
        size_t p = 1; while (p < x) p <<= 1; return p;
    }
    static inline void cpu_relax() noexcept {
    #if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
    #else
        std::this_thread::yield();
    #endif
    }

    alignas(64) std::vector<Cell> buffer_;
    alignas(64) std::atomic<size_t> enqueue_pos_{0};
    alignas(64) std::atomic<size_t> dequeue_pos_{0};
    size_t capacity_ = 0;
    size_t mask_ = 0;

public:
    explicit MpmcQueue(size_t capacity_hint = 1024) {
        capacity_ = round_up_pow2(capacity_hint ? capacity_hint : 1024);
        mask_ = capacity_ - 1;
        buffer_.resize(capacity_);
        for (size_t i = 0; i < capacity_; ++i) {
            buffer_[i].seq.store(i, std::memory_order_relaxed);
        }
        enqueue_pos_.store(0, std::memory_order_relaxed);
        dequeue_pos_.store(0, std::memory_order_relaxed);
    }

    void push(T&& v) {
        // 忙等直到成功，保持 void 接口语义
        while (!try_push(std::move(v))) cpu_relax();
    }

    bool try_push(T&& v) {
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            Cell& c = buffer_[pos & mask_];
            size_t seq = c.seq.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed)) {
                    // 抢到槽位后才移动构造，失败时不会消耗 v
                    c.data.emplace(std::move(v));
                    c.seq.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                // 满
                return false;
            } else {
                // 跟进位置再试
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
    }

    bool try_pop(T& out) {
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            Cell& c = buffer_[pos & mask_];
            size_t seq = c.seq.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
            if (diff == 0) {
                if (dequeue_pos_.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed)) {
                    out = std::move(*(c.data));
                    c.data.reset();
                    c.seq.store(pos + capacity_, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                // 空
                return false;
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
    }

    std::deque<T> drain_all() {
        std::deque<T> out;
        T tmp;
        while (try_pop(tmp)) {
            out.emplace_back(std::move(tmp));
        }
        return out;
    }

    bool empty() const {
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        const Cell& c = buffer_[pos & mask_];
        size_t seq = c.seq.load(std::memory_order_acquire);
        return static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1) < 0;
    }

    size_t capacity() const { return capacity_; }

    MpmcQueue(const MpmcQueue&) = delete;
    MpmcQueue& operator=(const MpmcQueue&) = delete;
};
