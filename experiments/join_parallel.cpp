// falcon_join.cpp
#include <bits/stdc++.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <future>
#include <atomic>
#include <iomanip>

#include "pgm/pgm_index.hpp"     
#include "FALCON/Falcon.hpp"     
#include "utils/include.hpp"     
#include "utils/utils.hpp"

using Key = uint64_t;
using Clock = std::chrono::high_resolution_clock;
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"

struct ProbeSpec {
    // 0=point, 1=range
    uint8_t is_range;
    uint64_t len;   // 对于 point = 1；对于 range = 区间内点数（用于从 queries 序列切分）
    // 在 queries 中：若 is_range=0，则取 1 个值作为点；若 is_range=1，则取 len 个值，lo=首，hi=尾
};

struct Stats {
    double avg_latency_ns = 0.0;  // 每个小查询（point/range）平均墙钟延迟
    double hit_ratio      = 0.0;  // 数据页缓存命中率
    uint64_t logical_ios = 0;   
    long long io_ns       = 0;    // I/O 花费时间（由 FALCON 统计）
    long long wall_ns     = 0;    // 端到端墙钟
    size_t height         = 0;    // PGM 高度
    uint64_t matched_total= 0;    // 所有 range/point 返回条目总数（辅助 sanity）
};

static std::vector<ProbeSpec> stitch_specs(const std::vector<uint8_t>& bitmap,
                                           const std::vector<uint64_t>& lens) {
    if (bitmap.size() != lens.size())
        throw std::runtime_error("bitmap and lengths size mismatch");
    std::vector<ProbeSpec> specs; specs.reserve(bitmap.size());
    for (size_t i=0;i<bitmap.size();++i) {
        ProbeSpec s{bitmap[i], lens[i]};
        specs.emplace_back(s);
    }
    return specs;
}

// ---------- worker ----------
template <size_t EPS>
static void probe_worker(falcon::FalconPGM<Key, EPS, 4>* engine,
                         const std::vector<ProbeSpec>& specs,
                         const std::vector<Key>& queries,
                         size_t begin_spec, size_t end_spec,
                         std::atomic<uint64_t>& matched_acc) {
    constexpr size_t BATCH = 128;
    size_t s_idx = begin_spec;

    // 计算本线程起始的 queries 偏移
    size_t q_idx = 0;
    for (size_t i = 0; i < begin_spec; ++i) q_idx += specs[i].len;

    struct Pending {
        std::future<falcon::RangeResult> fut;
        size_t qL, qR;   // 该任务对应的探针键窗口 [qL, qR)
    };

    std::vector<Pending> pendings;
    pendings.reserve(BATCH);

    auto submit = [&](Key lo, Key hi, size_t qL, size_t qR) {
        Pending p;
        p.fut = engine->range_lookup(lo, hi);
        p.qL = qL; p.qR = qR;
        pendings.emplace_back(std::move(p));
    };

    auto drain = [&]() {
        uint64_t local = 0;
        for (auto &p : pendings) {
            auto rr = p.fut.get();                  // [lo,hi] 内的 B 键（升序）
            size_t i = p.qL, j = 0;
            const auto &ret_keys = rr.keys;
            size_t qn = p.qR - p.qL;
            while (i < p.qR && j < ret_keys.size()) {
                uint64_t qk = queries[i];
                uint64_t rk = ret_keys[j];
                if (rk < qk) { ++j; }
                else if (rk == qk) { ++local; ++i; ++j; }
                else { ++i; }
            }
        }
        matched_acc.fetch_add(local, std::memory_order_relaxed);
        pendings.clear();
    };

    while (s_idx < end_spec) {
        pendings.clear();
        size_t upto = std::min(end_spec, s_idx + BATCH);
        for (; s_idx < upto; ++s_idx) {
            const auto& sp = specs[s_idx];

            if (sp.is_range == 0) {
                // ---- 修复 #1：point 分段要处理 sp.len 个点 ----
                // 把这一段的 len 个点逐个作为 (k,k) 提交，并且 q_idx += sp.len
                size_t remaining = sp.len;
                size_t pos = q_idx;
                while (remaining > 0) {
                    // 分小批提交，避免一次性塞太多 future
                    size_t chunk = std::min(remaining, (size_t)BATCH - pendings.size());
                    for (size_t t = 0; t < chunk; ++t) {
                        Key k = queries[pos + t];
                        submit(k, k, pos + t, pos + t + 1);
                    }
                    remaining -= chunk;
                    pos       += chunk;

                    if (pendings.size() >= BATCH) drain();
                }
                q_idx += sp.len;  // 注意：一次性跨过整段
            } else {
                // ---- 修复 #2：range 分段用该段 min/max 作为窗 ----
                size_t qL = q_idx;
                size_t qR = q_idx + sp.len;

                Key lo = queries[qL], hi = lo;
                for (size_t t = qL + 1; t < qR; ++t) {
                    Key v = queries[t];
                    if (v < lo) lo = v;
                    if (v > hi) hi = v;
                }
                submit(lo, hi, qL, qR);
                q_idx = qR;
            }
        }
        if (!pendings.empty()) drain();
    }
}

// ---------- bench driver ----------
template <size_t EPS>
static Stats run_join_falcon(const std::vector<Key>& build_keys,   // B 表（已排序）
                             const std::string& datafile_B,        // B 的数据文件（与索引对齐的记录文件）
                             const std::vector<ProbeSpec>& specs,  // A 的探针描述
                             const std::vector<Key>& queries,      // A 的探针序列（点或区间端点）
                             int threads,
                             size_t mem_budget_bytes,
                             pgm::CachePolicy policy,
                             pgm::IOInterface io_iface,
                             bool use_odirect) {
    // 1) 构建 PGM 索引（内存）
    pgm::PGMIndex<Key, EPS> pgm_idx(build_keys);

    // 2) 打开数据文件（B），用于最后一跳页读取
    int flags = O_RDONLY;
    if (use_odirect) flags |= O_DIRECT;
    int fd = ::open(datafile_B.c_str(), flags);
    if (fd < 0) { perror("open B data"); throw std::runtime_error("open datafile failed"); }

    // 3) 构建 FALCON 引擎（内部含缓存与 I/O backend）
    falcon::FalconPGM<Key, EPS, 4> engine(
        pgm_idx,
        fd,
        io_iface,
        /*memory_budget_bytes=*/ mem_budget_bytes,
        /*cache_policy=*/ policy,
        /*cache_shards=*/ 1,
        /*max_pages_per_batch=*/ 256,
        /*max_wait_us=*/ 50,
        /*workers=*/ std::min(std::max(threads/8, 1), 16)
    );

    // 4) 多线程探针（把 specs 分片；queries 顺序按 specs->len 前缀和切）
    auto t0 = Clock::now();
    std::atomic<uint64_t> matched_total{0};
    std::vector<std::thread> ths;
    ths.reserve(threads);
    size_t per = specs.size() / threads;
    for (int t=0;t<threads;++t) {
        size_t L = t * per;
        size_t R = (t==threads-1) ? specs.size() : (t+1)*per;
        ths.emplace_back(probe_worker<EPS>, &engine, std::cref(specs), std::cref(queries), L, R, std::ref(matched_total));
    }
    for (auto& th : ths) th.join();
    auto t1 = Clock::now();

    // 5) 统计
    auto st = engine.stats();
    Stats s;
    s.wall_ns       = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    s.avg_latency_ns = double(s.wall_ns) / std::max<size_t>(1, queries.size());
    s.hit_ratio     = (st.cache_hits + st.cache_misses) ? double(st.cache_hits) / double(st.cache_hits + st.cache_misses) : 0.0;
    s.logical_ios  = st.logical_ios;
    s.io_ns         = st.io_ns;
    s.height        = pgm_idx.height();
    s.matched_total = matched_total.load();
    ::close(fd);
    return s;
}

// ---------- CLI 解析 & main ----------
// static void usage(const char* prog) {
//     std::cerr <<
//     "Usage:\n  " << prog << " \\\n"
//     "    --build-file <SOSD_data_of_B> \\\n"
//     "    --probe-bin <A_queries.bin> --probe-par <A_lengths.par> --probe-bitmap <A.bitmap> \\\n"
//     "    --datafile-B <B_records_file> \\\n"
//     "    --threads <N> --epsilon <E> --mem-mib <MiB> \\\n"
//     "    [--policy LRU|FIFO|LFU] [--io psync|libaio|uring] [--no-odirect] [--trials K] [--csv out.csv]\n";
// }

int main(int argc, char** argv) {
    std::string build_file = std::string(DATASETS) + "books_200M_uint64_unique";         
    std::string probe_bin, probe_par, probe_bitmap; 
    probe_bin = std::string(DATASETS) + "books_200M_uint64_unique.1Mtable.bin";
    probe_par = std::string(DATASETS) + "books_200M_uint64_unique.1Mtable.par";
    probe_bitmap = std::string(DATASETS) + "books_200M_uint64_unique.1Mtable.bitmap";
    std::string datafile_B = build_file;         
    int threads = 1;
    // size_t epsilon = 16;
    size_t mem_mib = 256;
    std::string policy_str = "LRU";
    std::string io_str = "uring";
    bool use_odirect = true;
    int trials = 5;
    std::string csv_out = "books-200M-join.csv";

    // 加载 B 的 key（注意：需与 datafile_B 中记录顺序一致）
    auto build_keys = load_binary<Key>(build_file, /*has_header=*/false);

    // 加载 A 的探针描述 + queries
    auto lens    = load_binary<uint64_t>(probe_par, false);
    auto bitmap  = load_binary<uint8_t>(probe_bitmap, false);
    auto queries = load_binary<Key>(probe_bin, false);
    auto specs   = stitch_specs(bitmap, lens);

    // 校验 queries 长度
    size_t expect = 0; for (auto& s : specs) expect += s.len;
    if (expect != queries.size())
        throw std::runtime_error("probe_bin size mismatch with par/bitmap");

    // 解析策略/接口
    pgm::CachePolicy policy = pgm::CachePolicy::LRU;
    if (policy_str=="FIFO") policy = pgm::CachePolicy::FIFO;
    else if (policy_str=="LFU") policy = pgm::CachePolicy::LFU;

    pgm::IOInterface iface = pgm::IO_URING;
    if (io_str=="psync") iface = pgm::PSYNC;
    else if (io_str=="libaio") iface = pgm::LIBAIO;
    else iface = pgm::IO_URING;

    // CSV
    std::ofstream ofs(csv_out, std::ios::out | std::ios::trunc);
    ofs << "threads,epsilon,avg_latency_ns,avg_walltime_s,avg_IOs,data_IO_time\n";
    ofs << std::fixed << std::setprecision(6);

    auto bench_once = [&](auto const_tag){
        constexpr size_t EPS = decltype(const_tag)::value;
        // size_t idx_est = 16ull * N_KEYS / (2*EPS);
        // size_t buf_budget = (MEM_BUDGET > idx_est) ? (MEM_BUDGET - idx_est) : 0;
        for (int t=0;t<trials;++t) {
            auto s = run_join_falcon<EPS>(
                build_keys, datafile_B, specs, queries,
                threads, mem_mib*1024ull*1024ull, policy, iface, use_odirect
            );
            std::cout << "[trial " << (t+1) << "/" << trials << "] "
                      << "eps=" << EPS
                      << " thr=" << threads
                      << " hit=" << s.hit_ratio
                      << " pIOs=" << s.logical_ios
                      << " io_ms=" << (s.io_ns/1e6)
                      << " wall_s=" << (s.wall_ns/1e9)
                      << " H=" << s.height
                      << " matched=" << s.matched_total
                      << std::endl;

            ofs << threads << "," 
                << EPS << "," 
                << s.avg_latency_ns << "," 
                << (s.wall_ns/1e9) << "," 
                << s.logical_ios << "," 
                << s.io_ns << "\n";
            ofs.flush();
        }
    };
    for (auto epsilon : {16}){      // 8, 12, 16, 20, 24, 32, 48, 64, 128
        switch (epsilon) {
            case 8:   bench_once(std::integral_constant<size_t,8>{}); break;
            case 12:  bench_once(std::integral_constant<size_t,12>{}); break;
            case 16:  bench_once(std::integral_constant<size_t,16>{}); break;
            case 20:  bench_once(std::integral_constant<size_t,20>{}); break;
            case 24:  bench_once(std::integral_constant<size_t,24>{}); break;
            case 32:  bench_once(std::integral_constant<size_t,32>{}); break;
            case 48:  bench_once(std::integral_constant<size_t,48>{}); break;
            case 64:  bench_once(std::integral_constant<size_t,64>{}); break;
            case 128: bench_once(std::integral_constant<size_t,128>{}); break;
            default:  bench_once(std::integral_constant<size_t,16>{}); break;
        }
    }
    ofs.close();
    return 0;
}
