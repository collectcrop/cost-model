#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "utils/include.hpp"
#include "utils/utils.hpp"
#include "IO/io_interface.hpp"
#include "IO/SyncInterface.hpp"
#include "IO/LibaioInterface.hpp"
#include "IO/IOuringInterface.hpp"
#include "pgm/pgm_index.hpp"

using Key = uint64_t;
using Clock = std::chrono::high_resolution_clock;

struct BenchStat {
    uint64_t queries = 0;
    uint64_t found = 0;
    uint64_t logical_ios = 0;
    uint64_t physical_ios = 0;
    uint64_t bytes = 0;
    uint64_t ns = 0;
};
static inline void accumulate(BenchStat& b, const pgm::IOResult& io) {
    b.logical_ios  += io.logical_ios;
    b.physical_ios += io.physical_ios;
    b.bytes        += io.bytes;
    b.ns           += io.ns;
}
enum class Strategy { AllAtOnce, OneByOne };
enum class IOType { PSYNC, LIBAIO, URING };

#ifndef EPSILON
#define EPSILON 64
#endif

struct Args {
    std::string data_path;
    std::string query_path;     // optional
    IOType io = IOType::URING;
    Strategy strat = Strategy::AllAtOnce;
    int threads = 1;
    bool direct = true;
    size_t qcount = 1'000'000;  // used if no query file
};

static void usage(const char* prog) {
    std::cerr
      << "Usage: " << prog
      << " --data <books_200MB.bin> [--queries <queries.bin>]\n"
      << "       [--io psync|libaio|uring] [--strategy all|one]\n"
      << "       [--threads N] [--direct 0|1] [--qcount Q]\n";
}
static bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&](const char* k) {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << k << "\n"; return false; }
            return true;
        };

        if (s == "--data") {
            if (!need("--data")) return false;
            a.data_path = argv[++i];
        } else if (s == "--queries") {
            if (!need("--queries")) return false;
            a.query_path = argv[++i];
        } else if (s == "--io") {
            if (!need("--io")) return false;
            std::string v = argv[++i];
            if (v == "psync") a.io = IOType::PSYNC;
            else if (v == "libaio") a.io = IOType::LIBAIO;
            else if (v == "uring") a.io = IOType::URING;
            else { std::cerr << "Bad --io\n"; return false; }
        } else if (s == "--strategy") {
            if (!need("--strategy")) return false;
            std::string v = argv[++i];
            if (v == "all") a.strat = Strategy::AllAtOnce;
            else if (v == "one") a.strat = Strategy::OneByOne;
            else { std::cerr << "Bad --strategy\n"; return false; }
        } else if (s == "--threads") {
            if (!need("--threads")) return false;
            a.threads = std::stoi(argv[++i]);
        } else if (s == "--direct") {
            if (!need("--direct")) return false;
            a.direct = (std::stoi(argv[++i]) != 0);
        } else if (s == "--qcount") {
            if (!need("--qcount")) return false;
            a.qcount = (size_t)std::stoull(argv[++i]);
        } else {
            std::cerr << "Unknown arg: " << s << "\n";
            return false;
        }
    }

    if (a.data_path.empty()) {
        std::cerr << "--data is required.\n";
        return false;
    }
    return true;
}


static inline bool page_binary_search(const pgm::Page& pg, Key key) {
    if (!pg.data || pg.valid_len < sizeof(pgm::Record)) return false;
    auto* recs = reinterpret_cast<const pgm::Record*>(pg.data.get());
    size_t cnt = pg.valid_len / sizeof(pgm::Record);
    // quick reject
    if (recs[0].key > key || recs[cnt - 1].key < key) return false;

    size_t l = 0, r = cnt - 1;
    while (l <= r) {
        size_t m = l + ((r - l) >> 1);
        Key k = recs[m].key;
        if (k < key) l = m + 1;
        else if (k > key) {
            if (m == 0) return false;
            r = m - 1;
        } else return true;
    }
    return false;
}

template<class IndexT>
static bool lookup_all_at_once(IOInterface& io, const IndexT& idx, Key key, BenchStat& st) {
    auto ap = idx.search(key);
    size_t page_lo = ap.lo / pgm::ITEM_PER_PAGE;
    size_t page_hi = ap.hi / pgm::ITEM_PER_PAGE;
    if (page_hi < page_lo) page_hi = page_lo;
    size_t len = page_hi - page_lo + 1;

    auto [pages, r] = io.triggerIO(page_lo, len);   // one big sequential read
    accumulate(st, r);

    // early stop with page boundary checks
    for (size_t i = 0; i < pages.size(); ++i) {
        const auto& pg = pages[i];
        if (!pg.data || pg.valid_len == 0) continue;
        auto* recs = reinterpret_cast<const pgm::Record*>(pg.data.get());
        size_t cnt = pg.valid_len / sizeof(pgm::Record);
        if (cnt == 0) continue;

        if (recs[cnt - 1].key < key) continue;
        if (recs[0].key > key) break;
        if (page_binary_search(pg, key)) return true;
    }
    return false;
}

template<class IndexT>
static bool lookup_one_by_one(IOInterface& io, const IndexT& idx, Key key, BenchStat& st) {
    auto ap = idx.search(key);
    size_t page_lo = ap.lo / pgm::ITEM_PER_PAGE;
    size_t page_hi = ap.hi / pgm::ITEM_PER_PAGE;
    if (page_hi < page_lo) page_hi = page_lo;

    // Left-to-right probing: page_lo, page_lo+1, ..., page_hi
    for (size_t cur = page_lo; cur <= page_hi; ++cur) {
        auto [pg, r] = io.triggerIO(cur);
        accumulate(st, r);

        if (!pg.data || pg.valid_len == 0) continue;

        auto* recs = reinterpret_cast<const pgm::Record*>(pg.data.get());
        size_t cnt = pg.valid_len / sizeof(pgm::Record);
        if (cnt == 0) continue;

        // Early prune:
        // If key is smaller than the first key of current page, it cannot appear in later pages.
        if (key < recs[0].key) return false;

        // If key is larger than the last key, it might be in later pages; keep scanning.
        if (key > recs[cnt - 1].key) continue;

        // Key is within [first,last] of this page -> binary search.
        if (page_binary_search(pg, key)) return true;

        // In-range but not found => absent
        return false;
    }

    return false;
}

static int open_data_fd(const std::string& path, bool direct) {
    int flags = O_RDONLY;
#ifdef O_DIRECT
    if (direct) flags |= O_DIRECT;
#endif
    int fd = ::open(path.c_str(), flags);
    if (fd < 0 && direct) {
        // fallback
        fd = ::open(path.c_str(), O_RDONLY);
    }
    if (fd < 0) throw std::runtime_error("open fd failed");
    return fd;
}

static std::unique_ptr<IOInterface> make_io(IOType t, int fd) {
    if (t == IOType::PSYNC) return std::make_unique<SyncInterface>(fd);
    if (t == IOType::LIBAIO) return std::make_unique<LibaioInterface>(fd);
    return std::make_unique<IoUringInterface>(fd);
}
int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) { usage(argv[0]); return 1; }

    std::cerr << "Loading keys...\n";
    auto keys = load_binary<uint64_t>(args.data_path);
    std::cerr << "Keys: " << keys.size() << "\n";

    using IndexT = pgm::PGMIndex<Key, EPSILON, 4, float>;
    std::cerr << "Building PGM (EPS=" << EPSILON << ")...\n";
    IndexT index(keys.begin(), keys.end());
    std::cerr << "Index size(bytes): " << index.size_in_bytes() << "\n";

    auto queries = load_queries(args.query_path);
    std::cerr << "Queries: " << queries.size() << "\n";

    std::atomic<size_t> next{0};
    std::vector<std::thread> ths;
    std::vector<BenchStat> per(args.threads);

    auto wall0 = Clock::now();

    for (int t = 0; t < args.threads; ++t) {
        ths.emplace_back([&, t] {
            int fd = open_data_fd(args.data_path, args.direct);
            auto io = make_io(args.io, fd);

            BenchStat st;
            while (true) {
                size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= queries.size()) break;
                Key q = queries[i];
                bool ok = false;

                if (args.strat == Strategy::AllAtOnce) ok = lookup_all_at_once(*io, index, q, st);
                else ok = lookup_one_by_one(*io, index, q, st);

                st.queries++;
                if (ok) st.found++;
            }

            ::close(fd);
            per[t] = st;
        });
    }

    for (auto& th : ths) th.join();

    auto wall1 = Clock::now();
    uint64_t wall_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(wall1 - wall0).count();

    BenchStat agg;
    for (auto& s : per) {
        agg.queries += s.queries;
        agg.found += s.found;
        agg.logical_ios += s.logical_ios;
        agg.physical_ios += s.physical_ios;
        agg.bytes += s.bytes;
        agg.ns += s.ns;
    }
    std::cout << "agg queries:" << agg.queries << std::endl;
    double sec = (double)wall_ns / 1e9;
    double qps = agg.queries / sec;

    std::cout << "=== RESULT ===\n";
    std::cout << "strategy=" << (args.strat == Strategy::AllAtOnce ? "all" : "one")
              << " io=" << (args.io == IOType::PSYNC ? "psync" : (args.io == IOType::LIBAIO ? "libaio" : "uring"))
              << " threads=" << args.threads
              << " direct=" << (args.direct ? 1 : 0) << "\n";
    std::cout << "queries=" << agg.queries << " found=" << agg.found
              << " hit_ratio=" << (agg.queries ? (double)agg.found / agg.queries : 0.0) << "\n";
    std::cout << "wall_s=" << sec << " qps=" << qps << "\n";
    std::cout << "logical_ios=" << agg.logical_ios
              << " physical_ios=" << agg.physical_ios << "\n";
    std::cout << "bytes=" << agg.bytes
              << " avg_io_ns=" << (agg.physical_ios ? (double)agg.ns / agg.physical_ios : 0.0) << "\n";
    return 0;
}