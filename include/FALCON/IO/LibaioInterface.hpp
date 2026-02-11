
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
#include <libaio.h>

#include "FALCON/utils/include.hpp"
#include "FALCON/IO/io_interface.hpp"
using timer = std::chrono::high_resolution_clock;

namespace falcon {
class LibaioInterface : public falcon::IOInterface {
public:
    explicit LibaioInterface(int fd_){
        this->fd = fd_;
    }
    virtual ~LibaioInterface() = default;
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
        std::shared_ptr<char[]> buf(reinterpret_cast<char*>(raw), [](char* p){ free(p); });

        off_t offset = static_cast<off_t>(index) * static_cast<off_t>(PAGE);

        auto t0 = timer::now();
        ssize_t br = aio_pread_sync(buf.get(), PAGE, offset);
        if (br < 0){
            throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
        }
        auto t1 = timer::now();
        long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        falcon::Page p;
        p.data = std::move(buf);
        p.valid_len = static_cast<size_t>(br >= 0 ? br : 0);

        res.ns = ns;
        res.bytes = br > 0 ? br : 0;
        res.physical_ios = 1;

        return { std::move(p), res };
    }


    std::pair<std::vector<falcon::Page>, falcon::IOResult> triggerIO(size_t index, size_t len) override {
        falcon::IOResult res;
        std::vector<falcon::Page> pages;
        const size_t total_bytes = falcon::PAGE_SIZE * len;
        // allocate aligned page buffer
        void* raw = nullptr;
        if (posix_memalign(&raw, falcon::PAGE_SIZE, total_bytes) != 0) {
            throw std::runtime_error("posix_memalign failed");
        }
        std::shared_ptr<char[]> buf(reinterpret_cast<char*>(raw), [](char* p){ free(p); });

        off_t offset = static_cast<off_t>(index) * static_cast<off_t>(falcon::PAGE_SIZE);

        auto t0 = timer::now();
        ssize_t br = aio_pread_sync(buf.get(), total_bytes, offset);
        if (br < 0){
            throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
        }
        
        auto t1 = timer::now();
        long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        // then separate each page from Aggregated page
        size_t pages_read = (br + falcon::PAGE_SIZE - 1) / falcon::PAGE_SIZE;
        for (size_t i = 0; i < pages_read; i++) {
            falcon::Page page;
            void* page_ptr = nullptr;
            if (posix_memalign(&page_ptr, falcon::PAGE_SIZE, falcon::PAGE_SIZE) != 0) {
                throw std::runtime_error("posix_memalign failed for sub-page");
            }
            page.data.reset(reinterpret_cast<char*>(page_ptr), [](char* p){ free(p); });

            size_t copy_size = std::min(static_cast<size_t>(br - i * falcon::PAGE_SIZE), (size_t)falcon::PAGE_SIZE);
            memcpy(page.data.get(), buf.get() + i * falcon::PAGE_SIZE, copy_size);

            page.valid_len = copy_size;
            pages.push_back(std::move(page));
        }

        res.ns = ns;
        res.bytes = br > 0 ? br : 0;
        res.physical_ios = 1;
        res.logical_ios = len;
        return { std::move(pages), res };
    }
    
private:
    // helper: perform a single libaio pread of `bytes` into `buf` at `offset`.
    // This sets up a temporary io_context, submits one request and waits for its completion.
    // On success returns number of bytes read (>=0). On error throws.
    ssize_t aio_pread_sync(void* buf, size_t bytes, off_t offset) {
    io_context_t ctx = 0;
    const int maxevents = 1;
    if (io_setup(1, &ctx) < 0) {
    throw std::runtime_error(std::string("io_setup failed: ") + std::strerror(errno));
    }


    struct iocb cb;
    struct iocb* cbs[1];
    io_prep_pread(&cb, fd, buf, bytes, offset);
    cbs[0] = &cb;


    // submit
    int submit_ret = io_submit(ctx, 1, cbs);
    if (submit_ret < 0) {
    io_destroy(ctx);
    throw std::runtime_error(std::string("io_submit failed: ") + std::strerror(-submit_ret));
    }


    // wait for completion
    struct io_event events[1];
    int got = io_getevents(ctx, 1, 1, events, nullptr);
    if (got < 0) {
    io_destroy(ctx);
    throw std::runtime_error(std::string("io_getevents failed: ") + std::strerror(errno));
    }


    // event.res contains result (number of bytes or negative errno)
    long res = events[0].res;
    long res2 = events[0].res2; // usually 0


    io_destroy(ctx);


    if (res < 0) {
        // convert to positive errno and throw
        int err = static_cast<int>(-res);
        throw std::runtime_error(std::string("aio read failed: ") + std::strerror(err));
    }
    (void)res2;
    return static_cast<ssize_t>(res);
    }
};
}
