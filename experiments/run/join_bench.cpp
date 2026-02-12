// falcon_join.cpp
#include <bits/stdc++.h>
// #include <fcntl.h>
// #include <unistd.h>
// #include <thread>
// #include <future>
// #include <atomic>
// #include <iomanip>

#include "FALCON/pgm/pgm_index.hpp"     
#include "FALCON/Falcon.hpp"     
#include "FALCON/utils/include.hpp"     
#include "FALCON/utils/utils.hpp"
#include "./config.hpp"

using Key = uint64_t;
using Clock = std::chrono::high_resolution_clock;

struct ProbeSpec {
    // 0=point, 1=range
    uint8_t is_range;
    uint64_t len;   // point = 1；range = num of queries in a range
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

template <size_t EPS>
static void probe_worker(falcon::FalconPGM<Key, EPS, 4>* engine,
                         const std::vector<ProbeSpec>& specs,
                         const std::vector<Key>& queries,
                         size_t begin_spec, size_t end_spec,
                         std::atomic<uint64_t>& matched_acc) {
    constexpr size_t BATCH = 128;
    size_t s_idx = begin_spec;

    // calculate start positions of queries assigned to current thread
    size_t q_idx = 0;
    for (size_t i = 0; i < begin_spec; ++i) q_idx += specs[i].len;

    struct Pending {
        std::future<falcon::RangeResult> fut;
        size_t qL, qR;   
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
            auto rr = p.fut.get();                 
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
                // point query, coping with `len` nums of point probes 
                size_t remaining = sp.len;
                size_t pos = q_idx;
                while (remaining > 0) {
                    size_t chunk = std::min(remaining, (size_t)BATCH - pendings.size());
                    for (size_t t = 0; t < chunk; ++t) {
                        Key k = queries[pos + t];
                        submit(k, k, pos + t, pos + t + 1);
                    }
                    remaining -= chunk;
                    pos       += chunk;

                    if (pendings.size() >= BATCH) drain();
                }
                q_idx += sp.len;  
            } else {
                // range query
                size_t qL = q_idx;
                size_t qR = q_idx + sp.len;

                Key lo = queries[qL], hi = lo;
                // find the minimum and maximum key in the range
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
BenchmarkResult run_join_falcon(const std::vector<Key>& build_keys,   // sorted table B
                             const std::string& datafile_B,        // filename of B
                             const std::vector<ProbeSpec>& specs,  // Probe spec of A
                             std::vector<Key>& queries,      // table A
                             int threads,
                             size_t mem_budget_bytes,
                             falcon::CachePolicy policy,
                             falcon::IOInterfaceType io_iface,
                             bool use_odirect) {
    // 1) construct PGM
    pgm::PGMIndex<Key, EPS> pgm_idx(build_keys);

    // 2) open data file B
    int flags = O_RDONLY;
    if (use_odirect) flags |= O_DIRECT;
    int fd = ::open(datafile_B.c_str(), flags);
    if (fd < 0) { perror("open B data"); throw std::runtime_error("open datafile failed"); }

    // 3) initialize FALCON
    falcon::FalconPGM<Key, EPS, 4> engine(
        pgm_idx,
        fd,
        io_iface,
        /*memory_budget_bytes=*/ mem_budget_bytes,
        /*cache_policy=*/ policy,
        /*cache_shards=*/ 1,
        /*max_pages_per_batch=*/ 256,
        /*max_wait_us=*/ 50,
        /*workers=*/ std::max(threads/16, 1)
    );

    // 4) multi-thread probe
    auto t0 = Clock::now();
    sort(queries.begin(), queries.end());
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

    auto st = engine.stats();
    double hit_ratio = 0.0;
    auto hm = st.cache_hits + st.cache_misses;
    if (hm) hit_ratio = double(st.cache_hits) / double(hm);

    BenchmarkResult s;
    s.total_time       = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    s.avg_lat          = double(s.total_time) / std::max<size_t>(1, queries.size());
    s.hit_ratio        = hit_ratio;
    s.data_IOs         = st.logical_ios;
    s.height           = pgm_idx.height();
    ::close(fd);
    return s;
}


void run_join(JoinConfig &cfg){
    std::string build_file = falcon::DATASETS + cfg.dataset;         
    std::string probe_bin, probe_par, probe_bitmap; 
    probe_bin = falcon::DATASETS + cfg.query_file + std::string(".bin");
    probe_par = falcon::DATASETS + cfg.query_file + std::string(".par");
    probe_bitmap = falcon::DATASETS + cfg.query_file + std::string(".bitmap");
    std::string datafile_B = build_file;         
    // load keys of table B
    auto build_keys = load_binary<Key>(build_file, /*has_header=*/false);

    // load keys of table A and its probing sequence
    auto lens    = load_binary<uint64_t>(probe_par, false);
    auto bitmap  = load_binary<uint8_t>(probe_bitmap, false);
    auto queries = load_binary<Key>(probe_bin, false);
    auto specs   = stitch_specs(bitmap, lens);

    // check
    size_t expect = 0; for (auto& s : specs) expect += s.len;
    if (expect != queries.size())
        throw std::runtime_error("probe_bin size mismatch with par/bitmap");

    falcon::CachePolicy policy = cfg.policy;
    int threads = 1;
    size_t mem_mib = cfg.memory_mb;
    auto bench_once = [&](auto const_tag){
        constexpr size_t EPS = decltype(const_tag)::value;
        auto result = run_join_falcon<EPS>(
            build_keys, datafile_B, specs, queries,
            threads, mem_mib*1024ull*1024ull, policy, falcon::IO_URING, true
        );
        std::cout << "[Threads=1] ε=" << 16 << ", M=" << cfg.memory_mb  
                  << ", avg latency=" << result.avg_lat << " ns"
                  << ", hit ratio=" << result.hit_ratio
                  << ", total wall time=" << result.total_time / 1e9 << " s"
                  << ", data IOs=" << result.data_IOs
                  << std::endl;
    };
    for (auto epsilon : {16}){      // 2,4,6,10,12,14,16,18,20,24,32,48,64,128
        switch (epsilon) {
            case 2:   bench_once(std::integral_constant<size_t,2>{}); break;
            case 4:   bench_once(std::integral_constant<size_t,4>{}); break;
            case 6:   bench_once(std::integral_constant<size_t,6>{}); break;
            case 8:   bench_once(std::integral_constant<size_t,8>{}); break;
            case 10:  bench_once(std::integral_constant<size_t,10>{}); break;
            case 12:  bench_once(std::integral_constant<size_t,12>{}); break;
            case 14:  bench_once(std::integral_constant<size_t,14>{}); break;
            case 16:  bench_once(std::integral_constant<size_t,16>{}); break;
            case 18:  bench_once(std::integral_constant<size_t,18>{}); break;
            case 20:  bench_once(std::integral_constant<size_t,20>{}); break;
            case 24:  bench_once(std::integral_constant<size_t,24>{}); break;
            case 32:  bench_once(std::integral_constant<size_t,32>{}); break;
            case 48:  bench_once(std::integral_constant<size_t,48>{}); break;
            case 64:  bench_once(std::integral_constant<size_t,64>{}); break;
            case 128: bench_once(std::integral_constant<size_t,128>{}); break;
            default:  bench_once(std::integral_constant<size_t,16>{}); break;
        }
    }
}


// int main(int argc, char** argv) {
//     std::string build_file = falcon::DATASETS + std::string("books_200M_uint64_unique");         
//     std::string probe_bin, probe_par, probe_bitmap; 
//     probe_bin = falcon::DATASETS + std::string("books_200M_uint64_unique.1Mtable5.bin");
//     probe_par = falcon::DATASETS + std::string("books_200M_uint64_unique.1Mtable5.par");
//     probe_bitmap = falcon::DATASETS + std::string("books_200M_uint64_unique.1Mtable5.bitmap");
//     std::string datafile_B = build_file;         
//     int threads = 1;
//     // size_t epsilon = 16;
//     size_t mem_mib = 256;
//     std::string policy_str = "LRU";
//     std::string io_str = "uring";
//     bool use_odirect = true;
//     int trials = 10;
//     std::string csv_out = "books-200M-join.csv";

//     // 加载 B 的 key（注意：需与 datafile_B 中记录顺序一致）
//     auto build_keys = load_binary<Key>(build_file, /*has_header=*/false);

//     // 加载 A 的探针描述 + queries
//     auto lens    = load_binary<uint64_t>(probe_par, false);
//     auto bitmap  = load_binary<uint8_t>(probe_bitmap, false);
//     auto queries = load_binary<Key>(probe_bin, false);
//     auto specs   = stitch_specs(bitmap, lens);

//     // 校验 queries 长度
//     size_t expect = 0; for (auto& s : specs) expect += s.len;
//     if (expect != queries.size())
//         throw std::runtime_error("probe_bin size mismatch with par/bitmap");

//     // 解析策略/接口
//     falcon::CachePolicy policy = falcon::CachePolicy::NONE;
//     if (policy_str=="LRU") policy = falcon::CachePolicy::LRU;
//     else if (policy_str=="FIFO") policy = falcon::CachePolicy::FIFO;
//     else if (policy_str=="LFU") policy = falcon::CachePolicy::LFU;

//     falcon::IOInterfaceType iface = falcon::IO_URING;
//     if (io_str=="psync") iface = falcon::PSYNC;
//     else if (io_str=="libaio") iface = falcon::LIBAIO;
//     else iface = falcon::IO_URING;

//     // CSV
//     std::ofstream ofs(csv_out, std::ios::out | std::ios::trunc);
//     ofs << "threads,epsilon,avg_latency_ns,total_wall_time_s,avg_IOs,IO_time_s\n";
//     ofs << std::fixed << std::setprecision(6);

//     auto bench_once = [&](auto const_tag){
//         constexpr size_t EPS = decltype(const_tag)::value;
//         // size_t idx_est = 16ull * N_KEYS / (2*EPS);
//         // size_t buf_budget = (MEM_BUDGET > idx_est) ? (MEM_BUDGET - idx_est) : 0;
//         for (int t=0;t<trials;++t) {
//             auto s = run_join_falcon<EPS>(
//                 build_keys, datafile_B, specs, queries,
//                 threads, mem_mib*1024ull*1024ull, policy, iface, use_odirect
//             );
//             std::cout << "[trial " << (t+1) << "/" << trials << "] "
//                       << "eps=" << EPS
//                       << " thr=" << threads
//                       << " hit=" << s.hit_ratio
//                       << " pIOs=" << s.logical_ios
//                       << " io_s=" << (s.io_ns/1e9)
//                       << " wall_s=" << (s.wall_ns/1e9)
//                       << " H=" << s.height
//                       << " matched=" << s.matched_total
//                       << std::endl;

//             ofs << threads << "," 
//                 << EPS << "," 
//                 << s.avg_latency_ns << "," 
//                 << (s.wall_ns/1e9) << "," 
//                 << s.logical_ios << "," 
//                 << (s.io_ns/1e9) << "\n";
//             ofs.flush();
//         }
//     };
//     for (auto epsilon : {16}){      // 2,4,6,10,12,14,16,18,20,24,32,48,64,128
//         switch (epsilon) {
//             case 2:   bench_once(std::integral_constant<size_t,2>{}); break;
//             case 4:   bench_once(std::integral_constant<size_t,4>{}); break;
//             case 6:   bench_once(std::integral_constant<size_t,6>{}); break;
//             case 8:   bench_once(std::integral_constant<size_t,8>{}); break;
//             case 10:  bench_once(std::integral_constant<size_t,10>{}); break;
//             case 12:  bench_once(std::integral_constant<size_t,12>{}); break;
//             case 14:  bench_once(std::integral_constant<size_t,14>{}); break;
//             case 16:  bench_once(std::integral_constant<size_t,16>{}); break;
//             case 18:  bench_once(std::integral_constant<size_t,18>{}); break;
//             case 20:  bench_once(std::integral_constant<size_t,20>{}); break;
//             case 24:  bench_once(std::integral_constant<size_t,24>{}); break;
//             case 32:  bench_once(std::integral_constant<size_t,32>{}); break;
//             case 48:  bench_once(std::integral_constant<size_t,48>{}); break;
//             case 64:  bench_once(std::integral_constant<size_t,64>{}); break;
//             case 128: bench_once(std::integral_constant<size_t,128>{}); break;
//             default:  bench_once(std::integral_constant<size_t,16>{}); break;
//         }
//     }
//     ofs.close();
//     return 0;
// }
