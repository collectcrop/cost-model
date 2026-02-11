#pragma once
#include <vector>
#include <utility>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <chrono>
#include <algorithm>

#include <liburing.h>

#include "FALCON/utils/include.hpp"
#include "FALCON/IO/io_interface.hpp"
using timer = std::chrono::high_resolution_clock;

namespace falcon {
class IoUringInterface : public falcon::IOInterface {
public:
    explicit IoUringInterface(int fd_, unsigned queue_depth = 256) {
        this->fd = fd_;
        int ret = io_uring_queue_init(static_cast<unsigned>(queue_depth), &ring, 0);
        if (ret < 0) {
            throw std::runtime_error(std::string("io_uring_queue_init failed: ") + std::strerror(-ret));
        }
        // io_uring_params p{};
        // p.flags |= IORING_SETUP_SQPOLL;     // 开启内核SQ轮询
        // p.sq_thread_idle = 2000;            // 可配：空闲自旋ms
        // int ret = io_uring_queue_init_params(queue_depth, &ring, &p);
        // if (ret < 0) { /* 回退或报错 */ }
    }

    virtual ~IoUringInterface() {
        io_uring_queue_exit(&ring);
    }

    // single page read
    std::pair<falcon::Page, falcon::IOResult> triggerIO(size_t index) override {
        falcon::IOResult res;
        res.logical_ios = 1;
        const size_t PAGE = falcon::PAGE_SIZE;

        // allocate aligned page buffer
        void* raw = nullptr;
        if (posix_memalign(&raw, PAGE, PAGE) != 0) {
            throw std::runtime_error("posix_memalign failed");
        }
        // manage with free deleter
        std::shared_ptr<char[]> buf(reinterpret_cast<char*>(raw), [](char* p){ free(p); });

        off_t offset = static_cast<off_t>(index) * static_cast<off_t>(PAGE);

        auto t0 = timer::now();

        // prepare submission
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        if (!sqe) {
            throw std::runtime_error("io_uring_get_sqe returned null");
        }
        io_uring_prep_read(sqe, fd, buf.get(), PAGE, offset);
        // attach user data as the buffer ptr so we can access later (not strictly needed here)
        io_uring_sqe_set_data(sqe, buf.get());

        int submit_ret = io_uring_submit(&ring);
        if (submit_ret < 0) {
            throw std::runtime_error(std::string("io_uring_submit failed: ") + std::strerror(-submit_ret));
        }

        // wait for completion
        struct io_uring_cqe* cqe = nullptr;
        int wait_ret = io_uring_wait_cqe(&ring, &cqe);
        if (wait_ret < 0) {
            throw std::runtime_error(std::string("io_uring_wait_cqe failed: ") + std::strerror(-wait_ret));
        }

        long res_bytes = cqe->res;
        io_uring_cqe_seen(&ring, cqe);

        auto t1 = timer::now();
        long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        if (res_bytes < 0) {
            int err = static_cast<int>(-res_bytes);
            throw std::runtime_error(std::string("io_uring read failed: ") + std::strerror(err));
        }

        falcon::Page p;
        p.data = std::move(buf);
        p.valid_len = static_cast<size_t>(res_bytes >= 0 ? res_bytes : 0);

        res.ns = ns;
        res.bytes = res_bytes > 0 ? res_bytes : 0;
        res.physical_ios = 1;

        return { std::move(p), res };
    }

    // multi-page read (read len pages starting from index)
    std::pair<std::vector<falcon::Page>, falcon::IOResult> triggerIO(size_t index, size_t len) override {
        falcon::IOResult res;
        std::vector<falcon::Page> pages;

        const size_t total_bytes = falcon::PAGE_SIZE * len;
        // allocate an aggregated aligned buffer
        void* raw = nullptr;
        if (posix_memalign(&raw, falcon::PAGE_SIZE, total_bytes) != 0) {
            throw std::runtime_error("posix_memalign failed");
        }
        std::shared_ptr<char> buf(reinterpret_cast<char*>(raw), [](char* p){ free(p); });

        off_t offset = static_cast<off_t>(index) * static_cast<off_t>(falcon::PAGE_SIZE);

        auto t0 = timer::now();

        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        if (!sqe) {
            throw std::runtime_error("io_uring_get_sqe returned null");
        }
        io_uring_prep_read(sqe, fd, buf.get(), total_bytes, offset);
        io_uring_sqe_set_data(sqe, buf.get());

        int submit_ret = io_uring_submit(&ring);
        if (submit_ret < 0) {
            throw std::runtime_error(std::string("io_uring_submit failed: ") + std::strerror(-submit_ret));
        }

        struct io_uring_cqe* cqe = nullptr;
        int wait_ret = io_uring_wait_cqe(&ring, &cqe);
        if (wait_ret < 0) {
            throw std::runtime_error(std::string("io_uring_wait_cqe failed: ") + std::strerror(-wait_ret));
        }

        long br = cqe->res;
        io_uring_cqe_seen(&ring, cqe);

        auto t1 = timer::now();
        long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        if (br < 0) {
            int err = static_cast<int>(-br);
            throw std::runtime_error(std::string("io_uring read failed: ") + std::strerror(err));
        }

        // split into pages: allocate per-page buffers and copy
        size_t pages_read = (br + falcon::PAGE_SIZE - 1) / falcon::PAGE_SIZE;
        for (size_t i = 0; i < pages_read; ++i) {
            falcon::Page page;
            void* page_ptr = nullptr;
            if (posix_memalign(&page_ptr, falcon::PAGE_SIZE, falcon::PAGE_SIZE) != 0) {
                throw std::runtime_error("posix_memalign failed for sub-page");
            }
            page.data.reset(reinterpret_cast<char*>(page_ptr), [](char* p){ free(p); });

            size_t copy_size = std::min(static_cast<size_t>(br - i * falcon::PAGE_SIZE), (size_t)falcon::PAGE_SIZE);
            if (copy_size > 0) {
                memcpy(page.data.get(), buf.get() + i * falcon::PAGE_SIZE, copy_size);
            }
            page.valid_len = copy_size;
            pages.push_back(std::move(page));
        }

        res.ns = ns;
        res.bytes = br > 0 ? br : 0;
        res.physical_ios = 1; // aggregated IO
        res.logical_ios = len;

        return { std::move(pages), res };
    }

    std::pair<std::vector<falcon::Page>, falcon::IOResult>
    triggerIO_batch(const std::vector<size_t>& indices) override {
        falcon::IOResult stats{};
        std::vector<falcon::Page> out;
        out.resize(indices.size());
        if (indices.empty()) return { std::move(out), stats };

        // 1) 复制+排序（保留原位次，用于回填）
        struct Req { size_t page; size_t pos; };
        std::vector<Req> reqs;
        reqs.reserve(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            reqs.push_back({ indices[i], i });
        }
        // std::sort(reqs.begin(), reqs.end(), [](auto& a, auto& b){ return a.page < b.page; });

        // 2) 去重 + 相邻合并
        struct Range { size_t start; size_t len; std::vector<size_t> positions; };
        std::vector<Range> ranges;
        ranges.reserve(reqs.size());
        for (auto &r : reqs) {
            if (ranges.empty()) {
                ranges.push_back(Range{ r.page, 1, { r.pos } });
            } else {
                auto &last = ranges.back();
                size_t expect = last.start + last.len;
                if (r.page == expect) {               // 相邻页 → 合并
                    last.len++;
                    last.positions.push_back(r.pos);
                } else if (r.page < expect) {         // 重复页
                    last.positions.push_back(r.pos);
                } else {                               // 断裂 → 新片段
                    ranges.push_back(Range{ r.page, 1, { r.pos } });
                }
            }
        }

        // 3) 为每个片段分配聚合缓冲区并提交 SQE
        struct Pending {
            Range* range;
            std::shared_ptr<char> buf;
            size_t bytes;
        };
        std::vector<Pending> pendings;
        pendings.reserve(ranges.size());

        auto t0 = timer::now();

        for (auto &rg : ranges) {
            size_t bytes = rg.len * falcon::PAGE_SIZE;

            void* raw = nullptr;
            if (posix_memalign(&raw, falcon::PAGE_SIZE, bytes) != 0)
                throw std::runtime_error("posix_memalign failed");

            std::shared_ptr<char> agg(reinterpret_cast<char*>(raw), [](char* p){ free(p); });

            auto* sqe = io_uring_get_sqe(&ring);
            while (!sqe) {
                int sret = io_uring_submit(&ring); // flush
                if (sret < 0) throw std::runtime_error(std::string("io_uring_submit: ") + std::strerror(-sret));
                sqe = io_uring_get_sqe(&ring);
                // if (!sqe) throw std::runtime_error("io_uring_get_sqe null");
            }
            off_t off = static_cast<off_t>(rg.start) * static_cast<off_t>(falcon::PAGE_SIZE);
            io_uring_prep_read(sqe, fd, agg.get(), bytes, off);
            io_uring_sqe_set_data(sqe, &rg);

            pendings.push_back(Pending{ &rg, agg, bytes });
        }

        int submit_ret = io_uring_submit(&ring);
        if (submit_ret < 0) throw std::runtime_error(std::string("io_uring_submit: ") + std::strerror(-submit_ret));

        // 4) 等待所有片段完成并分发到 out（按调用顺序回填）
        size_t completed = 0;
        while (completed < pendings.size()) {
            io_uring_cqe* cqe = nullptr;
            int ret = io_uring_wait_cqe(&ring, &cqe);
            if (ret < 0) throw std::runtime_error(std::string("io_uring_wait_cqe: ") + std::strerror(-ret));

            auto* rg = reinterpret_cast<Range*>(io_uring_cqe_get_data(cqe));
            long br = cqe->res; // 读到的字节数或负 errno
            io_uring_cqe_seen(&ring, cqe);
            completed++;

            if (br < 0) {
                // 读失败：按请求位置填空页
                for (auto pos : rg->positions) out[pos] = falcon::Page{};
                continue;
            }

            // 找到对应 pending 以获取聚合缓冲
            auto it = std::find_if(pendings.begin(), pendings.end(), [&](const Pending& p){ return p.range == rg; });
            if (it == pendings.end()) continue;

            stats.bytes += br;
            stats.physical_ios += 1;

            size_t got_pages = (br + falcon::PAGE_SIZE - 1) / falcon::PAGE_SIZE;
            got_pages = std::min(got_pages, rg->len);

            // 对该片段内所有被请求的原始位次，切页复制回 out[pos]
            for (auto pos : rg->positions) {
                size_t page_idx = indices[pos];         // 原始页号
                size_t rel      = page_idx - rg->start; // 片段内相对偏移
                falcon::Page p;
                if (rel < got_pages) {
                    void* per = nullptr;
                    if (posix_memalign(&per, falcon::PAGE_SIZE, falcon::PAGE_SIZE) != 0)
                        throw std::runtime_error("posix_memalign sub failed");
                    std::shared_ptr<char[]> buf(reinterpret_cast<char*>(per), [](char* q){ free(q); });
                    size_t copy = std::min(static_cast<size_t>(br - rel*falcon::PAGE_SIZE), (size_t)falcon::PAGE_SIZE);
                    if ((long)copy > 0) {
                        std::memcpy(buf.get(), it->buf.get() + rel*falcon::PAGE_SIZE, copy);
                        p.data = std::move(buf);
                        p.valid_len = copy;
                    }
                }
                out[pos] = std::move(p);
            }
        }

        auto t1 = timer::now();
        stats.ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        stats.logical_ios = indices.size();

        return { std::move(out), stats };
    }
private:
    struct io_uring ring;
};

}
