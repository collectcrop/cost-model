// io_interface.hpp
#pragma once
#include <vector>
#include <utility>
#include <memory>
#include "FALCON/utils/include.hpp"
namespace falcon{

class IOInterface {
public:
    virtual ~IOInterface() = default;

    // single page read
    virtual std::pair<falcon::Page, falcon::IOResult> triggerIO(size_t index) = 0;
    // read multiple continuous pages [index, index+len)
    virtual std::pair<std::vector<falcon::Page>, falcon::IOResult> triggerIO(size_t index, size_t len) = 0;

    // batch read multiple (possibly non-continuous) pages
    virtual std::pair<std::vector<falcon::Page>, falcon::IOResult> triggerIO_batch(const std::vector<size_t>& indices) {
        std::vector<falcon::Page> out;
        falcon::IOResult agg{0,0};
        for (auto idx : indices) {
            auto [p, r] = triggerIO(idx);
            out.push_back(std::move(p));
            agg.ns += r.ns;
            agg.bytes += r.bytes;
        }
        return {std::move(out), agg};
    }

protected:
    int fd;
};

}
