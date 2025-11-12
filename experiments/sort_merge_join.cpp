// sort_merge_join_bench.cpp
// g++ sort_merge_join_bench.cpp -O3 -std=c++20 -lpthread -o sort_merge_join_bench

#include <bits/stdc++.h>
#include <chrono>
#include <iomanip>

using Key = uint64_t;
using Clock = std::chrono::high_resolution_clock;

// ======== 固定数据路径（按需修改） ========
static const std::string B_BUILD_FILE =
    "/mnt/home/zwshi/Datasets/SOSD/books_20M_uint64_unique";              // B 表 keys（二进制 uint64，已排序，通常 unique）
static const std::string A_PROBE_BIN =
    "/mnt/home/zwshi/Datasets/SOSD/books_20M_uint64_unique.100Ktable2.bin";
static const std::string A_PROBE_PAR =
    "/mnt/home/zwshi/Datasets/SOSD/books_20M_uint64_unique.100Ktable2.par";
static const std::string A_PROBE_BITMAP =
    "/mnt/home/zwshi/Datasets/SOSD/books_20M_uint64_unique.100Ktable2.bitmap";

// B 端是否 unique（*_unique 通常为真；如非唯一可改为 false）
static constexpr bool B_UNIQUE = true;

// ======== I/O 工具 ========
template<typename T>
std::vector<T> load_binary(const std::string& filename, bool has_header=false) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("open failed: " + filename);
    size_t n = 0;
    if (has_header) {
        uint64_t total=0; in.read(reinterpret_cast<char*>(&total), sizeof(total));
        n = static_cast<size_t>(total);
    } else {
        in.seekg(0, std::ios::end);
        auto bytes = static_cast<size_t>(in.tellg());
        in.seekg(0, std::ios::beg);
        n = bytes / sizeof(T);
    }
    std::vector<T> v(n);
    if (n) in.read(reinterpret_cast<char*>(v.data()), n*sizeof(T));
    if (!in) throw std::runtime_error("read failed: " + filename);
    return v;
}

struct ProbeSpec { uint8_t is_range; uint64_t len; };

static std::vector<ProbeSpec> stitch_specs(const std::vector<uint8_t>& bitmap,
                                           const std::vector<uint64_t>& lens) {
    if (bitmap.size() != lens.size())
        throw std::runtime_error("bitmap and lengths size mismatch");
    std::vector<ProbeSpec> v; v.reserve(bitmap.size());
    for (size_t i=0;i<bitmap.size();++i) v.push_back({bitmap[i], lens[i]});
    return v;
}

// 按 specs 把 A 端探针“逐键展开”
static std::vector<Key> expand_A_keys(const std::vector<ProbeSpec>& specs,
                                      const std::vector<Key>& queries) {
    std::vector<Key> A; A.reserve(queries.size());
    size_t qidx = 0;
    for (const auto& sp : specs) {
        for (uint64_t k=0; k<sp.len; ++k) {
            A.push_back(queries[qidx]);
            ++qidx;
        }
    }
    if (qidx != queries.size())
        throw std::runtime_error("expand_A_keys: queries not fully consumed");
    return A;
}

// ======== Sort-Merge Join（单线程） ========
struct SortMergeResult {
    uint64_t matches = 0;
    double sortA_s = 0.0;
    double merge_s = 0.0;
    double wall_s  = 0.0;
    size_t  a_sorted = 0;
};

static SortMergeResult sort_merge_join(std::vector<Key> A,  // 会排序副本
                                       const std::vector<Key>& B,
                                       bool B_unique) {
    SortMergeResult out{};

    // sort(A)
    auto t0 = Clock::now();
    std::sort(A.begin(), A.end());
    auto t1 = Clock::now();
    out.sortA_s = std::chrono::duration<double>(t1 - t0).count();
    out.a_sorted = A.size();

    // merge
    auto m0 = Clock::now();
    if (B_unique) {
        size_t i=0, j=0;
        while (i<A.size() && j<B.size()) {
            if (A[i] < B[j]) { ++i; }
            else if (A[i] > B[j]) { ++j; }
            else {
                Key k = A[i];
                size_t cntA=0;
                while (i<A.size() && A[i]==k) { ++cntA; ++i; }
                // B 的 runlen=1（unique）
                ++j;
                out.matches += cntA;
            }
        }
    } else {
        size_t i=0, j=0;
        while (i<A.size() && j<B.size()) {
            if (A[i] < B[j]) { ++i; }
            else if (A[i] > B[j]) { ++j; }
            else {
                Key k = A[i];
                size_t cntA=0, cntB=0;
                while (i<A.size() && A[i]==k) { ++cntA; ++i; }
                while (j<B.size() && B[j]==k) { ++cntB; ++j; }
                out.matches += static_cast<uint64_t>(cntA) * static_cast<uint64_t>(cntB);
            }
        }
    }
    auto m1 = Clock::now();
    out.merge_s = std::chrono::duration<double>(m1 - m0).count();

    out.wall_s = std::chrono::duration<double>(m1 - t0).count();
    return out;
}

int main() {
    // 读入数据
    auto B       = load_binary<Key>(B_BUILD_FILE, false);
    auto queries = load_binary<Key>(A_PROBE_BIN, false);
    auto lens    = load_binary<uint64_t>(A_PROBE_PAR, false);
    auto bitmap  = load_binary<uint8_t>(A_PROBE_BITMAP, false);
    if (std::accumulate(lens.begin(), lens.end(), uint64_t{0}) != queries.size())
        throw std::runtime_error("lens sum != queries size");
    auto specs = stitch_specs(bitmap, lens);
    auto A = expand_A_keys(specs, queries);

    // 跑 sort-merge
    auto r = sort_merge_join(A, B, /*B_unique=*/B_UNIQUE);

    // 打印并写 CSV
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[SortMergeJoin]\n"
              << "A=" << A.size() << ", B=" << B.size()
              << ", B_unique=" << (B_UNIQUE ? "true" : "false") << "\n"
              << "sortA_s=" << r.sortA_s << ", merge_s=" << r.merge_s
              << ", wall_s="  << r.wall_s  << ", matches=" << r.matches
              << ", A_sorted=" << r.a_sorted << std::endl;

    std::ofstream ofs("sort_merge_join_result.csv", std::ios::out | std::ios::trunc);
    ofs << "A_size,B_size,B_unique,sortA_s,merge_s,wall_s,matches,A_sorted\n";
    ofs << A.size() << "," << B.size() << "," << (B_UNIQUE?1:0) << ","
        << r.sortA_s << "," << r.merge_s << "," << r.wall_s << ","
        << r.matches << "," << r.a_sorted << "\n";
    ofs.close();
    return 0;
}
