// falcon_range_single.cpp
// g++ falcon_range_single.cpp -O3 -std=c++20 -lpthread -o falcon_range_single

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <future>
#include <thread>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#include "pgm/pgm_index.hpp"
#include "FALCON/Falcon.hpp"
#include "utils/include.hpp"

using KeyType = uint64_t;

#define DIRECTORY "/mnt/home/zwshi/learned-index/cost-model/experiments/"
#define DATASETS  "/mnt/home/zwshi/Datasets/SOSD/"

struct Record { uint64_t key; };

struct RangeQuery { uint64_t lo; uint64_t hi; };

struct BenchmarkResult {
    size_t epsilon;
    double time_ns;          // 单条查询平均时延（ns）
    double hit_ratio;        // 数据页缓存命中率
    time_t total_time;       // 全部查询墙钟（ns）
    time_t data_IO_time;     // 由 FALCON 统计的 I/O 时间（ns）
    size_t height;           // PGM 高度
    uint64_t physical_ios;   // 合并后的物理 I/O 次数
    uint64_t matched_total;  // 探针键命中数（口径A）
};

using timer = std::chrono::high_resolution_clock;


// ---------- 单线程、批式 range 查询（适配 FALCON；带“探针键口径A”统计） ----------
template <size_t Epsilon, size_t MemBudgetBytes>
BenchmarkResult benchmark_falcon(std::vector<KeyType> &data,
                                 const std::vector<RangeQuery>& ranges,
                                 const std::string& datafile_path,
                                 pgm::CachePolicy policy = pgm::CachePolicy::LRU,
                                 pgm::IOInterface io_iface = pgm::IO_URING,
                                 bool use_odirect = true,
                                 int workers = 1,
                                 // ↓ 可选：离散探针键流；若提供，则 matched_total 统计为“探针键命中数”
                                 const std::vector<KeyType>* probe_points = nullptr,
                                 // 每个 range 对应探针键窗口 [qL,qR)；若为空且只有一个 range，则默认 [0, probe_points->size())
                                 const std::vector<std::pair<size_t,size_t>>* range_windows = nullptr
                                 ) {
    // 1) 构建 PGM（内存）
    pgm::PGMIndex<KeyType, Epsilon> pgm_idx(data);

    // 2) 打开数据文件（外存记录，与 keys 顺序一致）
    int flags = O_RDONLY;
    if (use_odirect) flags |= O_DIRECT;
    int fd = ::open(datafile_path.c_str(), flags);
    if (fd < 0) { perror("open datafile"); throw std::runtime_error("open failed"); }

    // 3) 构建 FALCON 引擎（缓存+I/O 后端）
    falcon::FalconPGM<KeyType, Epsilon, 4> engine(
        pgm_idx,
        fd,
        io_iface,
        /*memory_budget_bytes=*/ MemBudgetBytes,
        /*cache_policy=*/ policy,
        /*cache_shards=*/ 1,
        /*max_pages_per_batch=*/ 256,
        /*max_wait_us=*/ 50,
        /*workers=*/ std::max(1, workers)
    );

    // 4) 执行查询：小批次提交；若提供 probe_points，则仅统计对应窗口内探针键的命中
    constexpr size_t BATCH = 128;
    uint64_t matched_total = 0;

    struct Pending {
        std::future<falcon::RangeResult> fut;
        size_t qL = 0, qR = 0;  // 对应的探针窗口（若无 probe_points 则忽略）
        bool has_window = false;
    };

    auto t0 = timer::now();
    std::vector<Pending> pendings; pendings.reserve(BATCH);

    auto submit = [&](KeyType lo, KeyType hi, size_t qL=0, size_t qR=0, bool hasW=false) {
        Pending p;
        p.fut = engine.range_lookup(lo, hi);
        p.qL = qL; p.qR = qR; p.has_window = hasW;
        pendings.emplace_back(std::move(p));
    };

    auto drain = [&]() {
        for (auto &p : pendings) {
            auto rr = p.fut.get(); // rr.keys 升序
            if (probe_points && p.has_window) {
                // 口径A：只统计该窗口内的探针键是否命中 B
                // 小窗口：逐键二分；大窗口可改 unordered_set 加速
                const auto& Q = *probe_points;
                for (size_t i = p.qL; i < p.qR; ++i) {
                    if (std::binary_search(rr.keys.begin(), rr.keys.end(), Q[i])) {
                        ++matched_total;
                    }
                }
            } else {
                // 若未提供探针键流，就回退到“payload 大小”统计（可选）
                matched_total += rr.keys.size();
            }
        }
        pendings.clear();
    };

    // 组装各 range 的提交
    size_t i = 0;
    while (i < ranges.size()) {
        pendings.clear();
        size_t j = std::min(ranges.size(), i + BATCH);
        for (; i < j; ++i) {
            const auto& rq = ranges[i];
            if (probe_points) {
                // 尝试获取该 range 的探针窗口
                size_t qL = 0, qR = 0; bool hasW = false;
                if (range_windows && i < range_windows->size()) {
                    qL = (*range_windows)[i].first;
                    qR = (*range_windows)[i].second;
                    hasW = true;
                } else if (ranges.size() == 1) {
                    // 只有一个 range 且给了 probe_points：默认全量窗口
                    qL = 0; qR = probe_points->size(); hasW = true;
                }
                submit(rq.lo, rq.hi, qL, qR, hasW);
            } else {
                submit(rq.lo, rq.hi);
            }
        }
        drain();
    }
    auto t1 = timer::now();

    // 5) 统计
    auto st = engine.stats();
    const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const auto n = std::max<size_t>(1, ranges.size());

    BenchmarkResult r;
    r.epsilon        = Epsilon;
    r.time_ns        = double(total_ns) / n;
    r.total_time     = total_ns;
    r.data_IO_time   = st.io_ns;
    r.hit_ratio      = (st.cache_hits + st.cache_misses)
                        ? double(st.cache_hits) / double(st.cache_hits + st.cache_misses) : 0.0;
    r.height         = pgm_idx.height();
    r.physical_ios   = st.logical_ios;
    r.matched_total  = matched_total;

    ::close(fd);
    return r;
}

int main() {
    std::string dataset  = "books";
    std::string filename = "books_200M_uint64_unique";
    std::string file     = std::string(DATASETS) + filename;

    // 这里延续你的“单个大区间”做法：用 raw_queries 的首尾构成一个 range
    std::string query_filename = "books_200M_uint64_unique.1Mtable3.bin";
    std::string query_file     = std::string(DATASETS) + query_filename;

    std::vector<KeyType> data        = load_binary<KeyType>(file, false);
    std::vector<KeyType> raw_queries = load_binary<KeyType>(query_file, false);
    raw_queries.resize(1000000);

    std::vector<RangeQuery> ranges;
    ranges.push_back({ raw_queries.front(), raw_queries.back() });

    // 若未来改为多区间 + 对应探针切片，可填充 range_windows，使每个区间只对自身探针做统计
    std::vector<std::pair<size_t,size_t>> range_windows; // 此例为空（单 range 时自动用 [0, |raw_queries|)）

    constexpr size_t MemoryBudget = 256ull * 1024 * 1024; // 40MiB
    const int trials = 10;

    auto out_csv = dataset + "-200M-range.csv";
    std::ofstream ofs(out_csv, std::ios::out | std::ios::trunc);
    ofs << "epsilon,threads,avg_latency_ns,total_wall_time_s,avg_IOs,IO_time_s\n";
    ofs << std::fixed << std::setprecision(6);

    for (pgm::CachePolicy policy : {pgm::CachePolicy::LRU}) {
        for (size_t epsilon : {16}) {
            BenchmarkResult result{};
            for (int t=0; t<trials; ++t) {
                switch (epsilon) {
                    case 16:
                        result = benchmark_falcon<16, MemoryBudget>(
                            data, ranges, file,
                            /*policy=*/policy,
                            /*io_iface=*/pgm::IO_URING,
                            /*use_odirect=*/true,
                            /*workers=*/1,
                            /*probe_points=*/&raw_queries,
                            /*range_windows=*/&range_windows // 空 -> 单 range 自动覆盖全窗口
                        );
                        break;
                    default: break;
                }

                std::cout << "ε=" << result.epsilon
                          << " | avg=" << result.time_ns << " ns"
                          << " | hit=" << result.hit_ratio
                          << " | IOs=" << result.physical_ios
                          << " | io_s=" << (result.data_IO_time/1e9)
                          << " | wall_s=" << (result.total_time/1e9)
                          << " | H=" << result.height
                          << " | matched=" << result.matched_total
                          << std::endl;

                ofs << epsilon << "," << 1 << ","
                    << result.time_ns << ","
                    << (result.total_time/1e9) << ","
                    << result.physical_ios << ","
                    << (result.data_IO_time/1e9) << "\n"; // ← 修复了原先漏逗号的问题
                ofs.flush();
            }
        }
    }
    ofs.close();
    return 0;
}
