#include <bits/stdc++.h>
#include <fcntl.h>
#include <unistd.h>
#include "utils/include.hpp"
#include "utils/utils.hpp"
#include "pgm/pgm_index.hpp"      // 只做“估计窗口”，不触发 IO
#include "FALCON/Falcon.hpp"           // 你的 FALCON 外层封装

using Key = uint64_t;
using timer = std::chrono::high_resolution_clock;

int main(int argc, char** argv) {
    // 路径可从命令行传入；否则用你的默认路径
    std::string data_path  = (argc > 1) ? argv[1] : "/mnt/home/zwshi/Datasets/SOSD/books_10M_uint64_unique";
    std::string query_path = (argc > 2) ? argv[2] : "/mnt/home/zwshi/Datasets/SOSD/books_10M_uint64_unique.query.bin";

    // 1) 载入数据与查询
    std::vector<Key> data   = load_data(data_path, 10'000'000);   // 10M
    std::vector<Key> querys = load_queries(query_path);           // 采样自 data，理应全部可命中
    if (data.empty() || querys.empty()) {
        std::cerr << "Load data/queries failed. paths:\n  data=" << data_path
                  << "\n  queries=" << query_path << std::endl;
        return 1;
    }

    // 2) 构建 PGM（只用于页窗口估计）
    constexpr size_t EPS = 8;
    // Mem/EpsRec 这两个模板参数只影响 PGM 内部结构，不影响 FALCON 的缓存/I/O
    using PGM = pgm::PGMIndex<Key, EPS>;
    PGM index(data);

    // 3) 打开数据文件（FALCON 读磁盘）
    int fd = ::open(data_path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open data"); return 2; }

    // 4) 构建 FALCON：1 worker + 1 shard（确定性），适当的批上限
    constexpr size_t MemoryBudget = 10ull << 20;     // 256 MiB 缓存
    falcon::FalconPGM<Key, EPS, 4> F(
        index,
        fd,
        pgm::IO_URING,
        /*memory_budget_bytes=*/ MemoryBudget,
        /*cache_policy=*/ pgm::CachePolicy::NONE,
        /*cache_shards=*/ 1,                // 为了结果稳定，先设 1
        /*max_pages_per_batch=*/ 256,       // 片段合并上限
        /*max_wait_us=*/ 1000000,         // 拉大时间窗，避免时间抖动影响（或改成你实现里的计数触发）
        /*workers=*/ 1
    );

    // 5) 一次提交一批 futures，让批量 I/O 生效（固定批大小以求确定性）
    const size_t BATCH = 1024;
    size_t found = 0;
    std::vector<Key> misses_sample; misses_sample.reserve(8);

    auto t0 = timer::now();
    std::vector<std::future<falcon::PointResult>> futs; futs.reserve(BATCH);
    for (size_t i = 0; i < querys.size(); ) {
        futs.clear();
        size_t j = std::min(querys.size(), i + BATCH);
        for (; i < j; ++i) futs.emplace_back(F.point_lookup(querys[i]));
        for (auto& f : futs) {
            auto r = f.get();
            if (r.found) ++found;
            else if (misses_sample.size() < 8) misses_sample.push_back(r.key);
        }
    }
    auto t1 = timer::now();

    // 6) 统计与断言
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    auto st = F.stats();
    double hit_ratio = 0.0;
    auto hm = st.cache_hits + st.cache_misses;
    if (hm) hit_ratio = double(st.cache_hits) / double(hm);

    std::cout << "Queries: " << querys.size()
              << ", Found: " << found
              << ", Expect: " << querys.size()
              << "\nAvg latency: " << (double)ns / querys.size() << " ns"
              << "\nCache hit ratio: " << hit_ratio
              << "\nPhysical IOs: " << st.physical_ios
              << ", IO bytes: " << st.io_bytes
              << ", IO time(ns): " << st.io_ns
              << std::endl;

    if (found != querys.size()) {
        std::cerr << "[ERROR] Some queries were not found! sample:";
        for (auto k : misses_sample) std::cerr << " " << k;
        std::cerr << std::endl;
        ::close(fd);
        return 3;
    }

    std::cout << "[PASS] All queries found." << std::endl;
    ::close(fd);
    return 0;
}
