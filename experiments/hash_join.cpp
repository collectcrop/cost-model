// hash_join.cpp
// g++ hash_join.cpp -O3 -std=c++20 -lpthread -o ./test -I ./include

#include <bits/stdc++.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>
#include "utils/utils.hpp"

using Key = uint64_t;
using Clock = std::chrono::high_resolution_clock;

// ======== 固定数据路径（按需修改） ========
static const std::string B_BUILD_FILE =
    "/mnt/home/zwshi/Datasets/SOSD/books_100M_uint64_unique";              // B 表 keys（二进制 uint64，已排序，通常 unique）
static const std::string A_PROBE_BIN =
    "/mnt/home/zwshi/Datasets/SOSD/books_100M_uint64_unique.1Mtable2.bin";

// B 端是否 unique（*_unique 通常为真；如非唯一可改为 false）
static constexpr bool B_UNIQUE = true;

// probe 阶段线程数（仅用于 Hash Join 的 probe；build 阶段单线程）
static int PROBE_THREADS = 1;

// ======== Hash Join ========
struct HashJoinResult {
    uint64_t matches = 0;  // 等值匹配条数（若 B 非 unique，则为频次相加）
    double   build_s = 0.0;
    double   probe_s = 0.0;
    double   wall_s  = 0.0;
    size_t   build_size = 0;
};

static HashJoinResult hash_join(const std::vector<Key>& A,
                                const std::vector<Key>& B,
                                bool B_unique,
                                int threads) {
    HashJoinResult out{};

    // Build
    auto t0 = Clock::now();
    if (B_unique) {
        std::unordered_set<Key> H;
        H.reserve(B.size() * 13 / 10);
        H.insert(B.begin(), B.end());
        out.build_size = H.size();
        auto t1 = Clock::now();
        out.build_s = std::chrono::duration<double>(t1 - t0).count();

        // Probe（并行分块）
        std::atomic<uint64_t> acc{0};
        auto worker = [&](size_t L, size_t R){
            uint64_t local=0;
            for (size_t i=L; i<R; ++i)
                if (H.find(A[i]) != H.end()) ++local;
            acc.fetch_add(local, std::memory_order_relaxed);
        };
        auto p0 = Clock::now();
        if (threads <= 1) worker(0, A.size());
        else {
            std::vector<std::thread> ths;
            size_t per = (A.size() + threads - 1) / threads;
            for (int t=0; t<threads; ++t) {
                size_t L = t*per, R = std::min(A.size(), (t+1)*per);
                if (L>=R) break;
                ths.emplace_back(worker, L, R);
            }
            for (auto& th : ths) th.join();
        }
        auto p1 = Clock::now();
        out.probe_s  = std::chrono::duration<double>(p1 - p0).count();
        out.matches  = acc.load();
        out.wall_s   = std::chrono::duration<double>(p1 - t0).count();
    } else {
        std::unordered_map<Key, uint32_t> H;
        H.reserve(B.size() * 13 / 10);
        for (auto k : B) ++H[k];
        out.build_size = H.size();
        auto t1 = Clock::now();
        out.build_s = std::chrono::duration<double>(t1 - t0).count();

        std::atomic<uint64_t> acc{0};
        auto worker = [&](size_t L, size_t R){
            uint64_t local=0;
            for (size_t i=L; i<R; ++i) {
                auto it = H.find(A[i]);
                if (it != H.end()) local += it->second;
            }
            acc.fetch_add(local, std::memory_order_relaxed);
        };
        auto p0 = Clock::now();
        if (threads <= 1) worker(0, A.size());
        else {
            std::vector<std::thread> ths;
            size_t per = (A.size() + threads - 1) / threads;
            for (int t=0; t<threads; ++t) {
                size_t L = t*per, R = std::min(A.size(), (t+1)*per);
                if (L>=R) break;
                ths.emplace_back(worker, L, R);
            }
            for (auto& th : ths) th.join();
        }
        auto p1 = Clock::now();
        out.probe_s  = std::chrono::duration<double>(p1 - p0).count();
        out.matches  = acc.load();
        out.wall_s   = std::chrono::duration<double>(p1 - t0).count();
    }
    return out;
}

int main() {
    // 读入数据
    auto B       = load_binary<Key>(B_BUILD_FILE, false);
    auto A =       load_binary<Key>(A_PROBE_BIN, false);
    // 跑 hash join
    auto r = hash_join(A, B, /*B_unique=*/B_UNIQUE, PROBE_THREADS);

    // 打印并写 CSV
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[HashJoin]\n"
              << "A=" << A.size() << ", B=" << B.size()
              << ", B_unique=" << (B_UNIQUE ? "true" : "false")
              << ", threads=" << PROBE_THREADS << "\n"
              << "build_s=" << r.build_s << ", probe_s=" << r.probe_s
              << ", wall_s="  << r.wall_s  << ", matches=" << r.matches
              << ", build_size=" << r.build_size << std::endl;

    std::ofstream ofs("hash_join_result.csv", std::ios::out | std::ios::trunc);
    ofs << "A_size,B_size,B_unique,threads,build_s,probe_s,wall_s,matches,build_size\n";
    ofs << A.size() << "," << B.size() << "," << (B_UNIQUE?1:0) << "," << PROBE_THREADS << ","
        << r.build_s << "," << r.probe_s << "," << r.wall_s << ","
        << r.matches << "," << r.build_size << "\n";
    ofs.close();
    return 0;
}
