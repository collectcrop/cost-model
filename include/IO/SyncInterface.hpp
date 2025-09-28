
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

#include "utils/include.hpp"
#include "IO/io_interface.hpp"
using timer = std::chrono::high_resolution_clock;

class SyncInterface : public IOInterface {
public:
    explicit SyncInterface(int fd_){
        this->fd = fd_;
    }
    virtual ~SyncInterface() = default;
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
        std::shared_ptr<char[]> buf(reinterpret_cast<char*>(raw), [](char* p){ free(p); });

        off_t offset = static_cast<off_t>(index) * static_cast<off_t>(PAGE);

        auto t0 = timer::now();
        ssize_t br;
        br = pread(fd, buf.get(), PAGE, offset);
        if (br < 0){
            throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
        }
        auto t1 = timer::now();
        long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        pgm::Page p;
        p.data = std::move(buf);
        p.valid_len = static_cast<size_t>(br >= 0 ? br : 0);

        res.ns = ns;
        res.bytes = br > 0 ? br : 0;
        res.physical_ios = 1;

        return { std::move(p), res };
    }


    std::pair<std::vector<pgm::Page>, pgm::IOResult> triggerIO(size_t index, size_t len) override {
        pgm::IOResult res;
        std::vector<pgm::Page> pages;
        // allocate aligned page buffer
        void* raw = nullptr;
        if (posix_memalign(&raw, pgm::PAGE_SIZE, pgm::PAGE_SIZE*len) != 0) {
            throw std::runtime_error("posix_memalign failed");
        }
        std::shared_ptr<char[]> buf(reinterpret_cast<char*>(raw), [](char* p){ free(p); });

        off_t offset = static_cast<off_t>(index) * static_cast<off_t>(pgm::PAGE_SIZE);

        auto t0 = timer::now();
        ssize_t br;
        
        br = pread(fd, buf.get(), pgm::PAGE_SIZE*len, offset);
        if (br < 0){
            throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
        }
        
        auto t1 = timer::now();
        long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        // then separate each page from Aggregated page
        size_t pages_read = (br + pgm::PAGE_SIZE - 1) / pgm::PAGE_SIZE;
        for (size_t i = 0; i < pages_read; i++) {
            pgm::Page page;
            void* page_ptr = nullptr;
            if (posix_memalign(&page_ptr, pgm::PAGE_SIZE, pgm::PAGE_SIZE) != 0) {
                throw std::runtime_error("posix_memalign failed for sub-page");
            }
            page.data.reset(reinterpret_cast<char*>(page_ptr), [](char* p){ free(p); });

            size_t copy_size = std::min(static_cast<size_t>(br - i * pgm::PAGE_SIZE), (size_t)pgm::PAGE_SIZE);
            memcpy(page.data.get(), buf.get() + i * pgm::PAGE_SIZE, copy_size);

            page.valid_len = copy_size;
            pages.push_back(std::move(page));
        }

        res.ns = ns;
        res.bytes = br > 0 ? br : 0;
        res.physical_ios = 1;
        res.logical_ios = len;
        return { std::move(pages), res };
    }
};