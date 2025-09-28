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

#include "utils/include.hpp"
#include "IO/io_interface.hpp"
using timer = std::chrono::high_resolution_clock;

// Note: link with -luring
class IoUringInterface : public IOInterface {
public:
    explicit IoUringInterface(int fd_, unsigned queue_depth = 256) {
        this->fd = fd_;
        int ret = io_uring_queue_init(static_cast<unsigned>(queue_depth), &ring, 0);
        if (ret < 0) {
            throw std::runtime_error(std::string("io_uring_queue_init failed: ") + std::strerror(-ret));
        }
    }

    virtual ~IoUringInterface() {
        io_uring_queue_exit(&ring);
    }

    // single page read
    std::pair<pgm::Page, pgm::IOResult> triggerIO(size_t index) override {
        pgm::IOResult res;
        res.logical_ios = 1;
        const size_t PAGE = pgm::PAGE_SIZE;

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

        pgm::Page p;
        p.data = std::move(buf);
        p.valid_len = static_cast<size_t>(res_bytes >= 0 ? res_bytes : 0);

        res.ns = ns;
        res.bytes = res_bytes > 0 ? res_bytes : 0;
        res.physical_ios = 1;

        return { std::move(p), res };
    }

    // multi-page read (read len pages starting from index)
    std::pair<std::vector<pgm::Page>, pgm::IOResult> triggerIO(size_t index, size_t len) override {
        pgm::IOResult res;
        std::vector<pgm::Page> pages;

        const size_t total_bytes = pgm::PAGE_SIZE * len;
        // allocate an aggregated aligned buffer
        void* raw = nullptr;
        if (posix_memalign(&raw, pgm::PAGE_SIZE, total_bytes) != 0) {
            throw std::runtime_error("posix_memalign failed");
        }
        std::shared_ptr<char> buf(reinterpret_cast<char*>(raw), [](char* p){ free(p); });

        off_t offset = static_cast<off_t>(index) * static_cast<off_t>(pgm::PAGE_SIZE);

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
        size_t pages_read = (br + pgm::PAGE_SIZE - 1) / pgm::PAGE_SIZE;
        for (size_t i = 0; i < pages_read; ++i) {
            pgm::Page page;
            void* page_ptr = nullptr;
            if (posix_memalign(&page_ptr, pgm::PAGE_SIZE, pgm::PAGE_SIZE) != 0) {
                throw std::runtime_error("posix_memalign failed for sub-page");
            }
            page.data.reset(reinterpret_cast<char*>(page_ptr), [](char* p){ free(p); });

            size_t copy_size = std::min(static_cast<size_t>(br - i * pgm::PAGE_SIZE), (size_t)pgm::PAGE_SIZE);
            if (copy_size > 0) {
                memcpy(page.data.get(), buf.get() + i * pgm::PAGE_SIZE, copy_size);
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

private:
    struct io_uring ring;
};
