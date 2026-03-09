// pgm_disk_range_parallel.cpp
#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <thread>
#include <atomic>
#include <fcntl.h>
#include <iomanip>
#include <stdexcept>
#include <CLI/CLI.hpp>
#include "FALCON/pgm/pgm_index.hpp"
#include "FALCON/utils/include.hpp"   
#include "FALCON/utils/utils.hpp"     

using KeyType = uint64_t;

struct BenchmarkResult {
    size_t   epsilon;
    double   avg_latency_ns;   
    double   hit_ratio;     
    long long wall_ns;         
    long long data_io_ns; 
    size_t   height;         
    uint64_t physical_ios;     
    uint64_t logical_ios;      
};


template <size_t Epsilon>
static void worker_range_pgm_disk(const pgm::PGMIndex<KeyType, Epsilon>& index,
                                  int fd,
                                  const std::vector<falcon::RangeQ>& queries,
                                  size_t begin,
                                  size_t end,
                                  std::atomic<uint64_t>& logical_ios,
                                  std::atomic<uint64_t>& physical_ios,
                                  std::atomic<long long>& io_time_ns,
                                  std::atomic<long long>& query_lat_ns_total)
{
    using Clock = std::chrono::high_resolution_clock;
    const size_t PAGE = falcon::PAGE_SIZE;

    if (begin >= end) {
        return;
    }

    size_t max_span_pages = 1;
    for (size_t i = begin; i < end; ++i) {
        const auto& q = queries[i];
        auto [plo, phi] = index.estimate_pages_for_range(q.lo, q.hi);
        if (phi >= plo) {
            size_t span = phi - plo + 1;
            if (span > max_span_pages) {
                max_span_pages = span;
            }
        }
    }

    size_t buf_bytes = max_span_pages * PAGE;
    void* raw = nullptr;
    if (posix_memalign(&raw, PAGE, buf_bytes) != 0) {
        throw std::runtime_error("posix_memalign failed");
    }
    char* buf = reinterpret_cast<char*>(raw);

    std::vector<KeyType> results;
    results.reserve(256);

    long long local_query_lat_ns = 0;

    for (size_t i = begin; i < end; ++i) {
        const auto& q = queries[i];

        auto q_t0 = Clock::now();

        auto [page_lo, page_hi] = index.estimate_pages_for_range(q.lo, q.hi);
        if (page_hi < page_lo) {
            auto q_t1 = Clock::now();
            local_query_lat_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(q_t1 - q_t0).count();
            continue;
        }

        size_t span_pages = page_hi - page_lo + 1;
        size_t read_bytes = span_pages * PAGE;
        off_t off = static_cast<off_t>(page_lo) * static_cast<off_t>(PAGE);

        auto io_t0 = Clock::now();
        ssize_t br = pread(fd, buf, read_bytes, off);
        auto io_t1 = Clock::now();

        if (br > 0) {
            long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(io_t1 - io_t0).count();
            io_time_ns.fetch_add(ns, std::memory_order_relaxed);
            logical_ios.fetch_add(span_pages, std::memory_order_relaxed);
            physical_ios.fetch_add(1, std::memory_order_relaxed);

            auto* recs = reinterpret_cast<falcon::Record*>(buf);
            size_t cnt = static_cast<size_t>(br) / sizeof(falcon::Record);
            if (cnt > 0) {
                auto* lb = std::lower_bound(
                    recs, recs + cnt, q.lo,
                    [](const falcon::Record& a, uint64_t k) { return a.key < k; }
                );
                auto* ub = std::upper_bound(
                    recs, recs + cnt, q.hi,
                    [](uint64_t k, const falcon::Record& a) { return k < a.key; }
                );

                results.clear();
                for (auto* cur = lb; cur < ub; ++cur) {
                    results.push_back(cur->key);
                }
                (void)results.size();
            }
        }

        auto q_t1 = Clock::now();
        local_query_lat_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(q_t1 - q_t0).count();
    }

    query_lat_ns_total.fetch_add(local_query_lat_ns, std::memory_order_relaxed);

    free(raw);
}

// ---- PGM-disk range benchmark driver ----
template <size_t Epsilon>
static BenchmarkResult bench_range_pgm_disk(const std::vector<KeyType>& data,
                                            const std::vector<falcon::RangeQ>& ranges,
                                            const std::string& datafile,
                                            int num_threads)
{
    using Clock = std::chrono::high_resolution_clock;

    pgm::PGMIndex<KeyType, Epsilon> index(data);

    int fd = ::open(datafile.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open data");
        std::exit(1);
    }

    std::atomic<uint64_t> logical_ios{0};
    std::atomic<uint64_t> physical_ios{0};
    std::atomic<long long> io_time_ns{0};
    std::atomic<long long> query_lat_ns_total{0};  
    std::vector<std::thread> ths;
    ths.reserve(num_threads);
    size_t per = ranges.size() / num_threads;

    auto t0 = Clock::now();
    for (int t = 0; t < num_threads; ++t) {
        size_t L = t * per;
        size_t R = (t == num_threads - 1) ? ranges.size() : (t + 1) * per;
        ths.emplace_back(worker_range_pgm_disk<Epsilon>,
                         std::cref(index),
                         fd,
                         std::cref(ranges),
                         L, R,
                         std::ref(logical_ios),
                         std::ref(physical_ios),
                         std::ref(io_time_ns),
                         std::ref(query_lat_ns_total));
    }
    for (auto& th : ths) th.join();
    auto t1 = Clock::now();

    long long wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    BenchmarkResult out{};
    out.epsilon        = Epsilon;
    out.wall_ns        = wall_ns;
    out.avg_latency_ns = double(query_lat_ns_total.load()) / std::max<size_t>(1, ranges.size());
    out.hit_ratio      = 0.0;
    out.data_io_ns     = io_time_ns.load();
    out.height         = index.height();
    out.logical_ios    = logical_ios.load();
    out.physical_ios   = physical_ios.load();

    ::close(fd);
    return out;
}

// int main(int argc, char** argv)
// {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0]
//                   << " <dataset_base> [n_keys]\n"
//                   << "  e.g. " << argv[0]
//                   << " fb_200M_uint64_unique 200000000\n";
//         return 1;
//     }

//     const std::string dataset  = argv[1];              
//     const int N_KEYS = (argc >= 3) ? std::stoi(argv[2])
//                                    : 200000000;        

//     const std::string datafile  = std::string(falcon::DATASETS) + dataset;
//     const std::string rangefile = std::string(falcon::DATASETS) + dataset + ".range.bin";

//     const size_t repeat  = 3;
//     const size_t EPSILON = 16;    

//     auto data   = load_data_pgm_safe<KeyType>(datafile, N_KEYS);
//     auto ranges = load_ranges_pgm_safe(rangefile);
//     if (data.empty() || ranges.empty()) {
//         std::cerr << "load data or ranges failed\n";
//         return 1;
//     }

//     std::ofstream csv(dataset + "_pgm_disk_range.csv",
//                       std::ios::out | std::ios::trunc);
//     if (!csv) {
//         std::cerr << "open csv failed\n";
//         return 1;
//     }

//     csv << "baseline,epsilon,threads,avg_latency_ns,wall_s,hit_ratio,"
//            "avg_IOs,physical_IOs,data_io_ns\n";
//     csv << std::fixed << std::setprecision(6);

//     for (size_t r = 0; r < repeat; ++r) {
//         for (int threads : {1,2,4,8,16,32,64,128,256,512,1024}) {
//             BenchmarkResult res = bench_range_pgm_disk<EPSILON>(data, ranges, datafile, threads);

//             std::cout << "[PGM-disk][T=" << threads << "] eps=" << res.epsilon
//                       << " | avg="  << res.avg_latency_ns << " ns"
//                       << " | wall=" << (res.wall_ns / 1e9) << " s"
//                       << " | logical_IOs="  << res.logical_ios
//                       << " | physical_IOs=" << res.physical_ios
//                       << " | io_ns="        << res.data_io_ns
//                       << std::endl;

//             csv << "PGM-disk," << res.epsilon << "," << threads << ","
//                 << res.avg_latency_ns << "," << (res.wall_ns / 1e9) << ","
//                 << res.hit_ratio << "," << res.logical_ios << ","
//                 << res.physical_ios << ","
//                 << res.data_io_ns << "\n";
//             csv.flush();
//         }
//     }

//     csv.close();
//     return 0;
// }

int main(int argc, char** argv) {
    CLI::App app{"PGM disk range benchmark"};

    std::string dataset_basename;
    uint64_t num_keys = 200000000;
    std::string baseline = "PGM-disk";
    std::string output_csv;
    size_t repeat = 3;

    app.add_option(
        "--dataset",
        dataset_basename,
        "Dataset basename, e.g. fb_200M_uint64_unique"
    )->required();

    app.add_option(
        "--keys",
        num_keys,
        "Number of keys to load"
    )->default_val(200000000);

    app.add_option(
        "--baseline",
        baseline,
        "Baseline name written to CSV"
    )->default_val("PGM-disk");

    app.add_option(
        "--repeat",
        repeat,
        "Number of repetitions for each thread count"
    )->default_val(3);

    app.add_option(
        "--output",
        output_csv,
        "Output CSV filename (default: <dataset>_pgm_disk_range.csv)"
    );

    CLI11_PARSE(app, argc, argv);

    const std::string datafile  = std::string(falcon::DATASETS) + dataset_basename;
    const std::string rangefile = std::string(falcon::DATASETS) + dataset_basename + ".range.bin";

    if (output_csv.empty()) {
        output_csv = dataset_basename + "_pgm_disk_range.csv";
    }

    constexpr size_t EPSILON = 16;
    const std::vector<int> thread_list = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    std::cout << "Loading data from: " << datafile << "\n";
    auto data = load_data_pgm_safe<KeyType>(datafile, num_keys);

    std::cout << "Loading range queries from: " << rangefile << "\n";
    auto ranges = load_ranges_pgm_safe(rangefile);

    if (data.empty() || ranges.empty()) {
        std::cerr << "load data or ranges failed\n";
        return 1;
    }

    std::ofstream csv(output_csv, std::ios::out | std::ios::trunc);
    if (!csv.is_open()) {
        std::cerr << "Failed to open output csv: " << output_csv << "\n";
        return 1;
    }

    csv << "baseline,epsilon,threads,run_id,avg_latency_ns,wall_s,hit_ratio,"
           "avg_IOs,physical_IOs,data_io_ns\n";
    csv << std::fixed << std::setprecision(6);

    for (size_t r = 0; r < repeat; ++r) {
        for (int threads : thread_list) {
            BenchmarkResult res = bench_range_pgm_disk<EPSILON>(data, ranges, datafile, threads);

            std::cout << "[" << baseline << "]"
                      << "[run=" << (r + 1) << "/" << repeat << "]"
                      << "[T=" << threads << "]"
                      << " eps=" << res.epsilon
                      << " | avg=" << res.avg_latency_ns << " ns"
                      << " | wall=" << (res.wall_ns / 1e9) << " s"
                      << " | logical_IOs=" << res.logical_ios
                      << " | physical_IOs=" << res.physical_ios
                      << " | io_ns=" << res.data_io_ns
                      << std::endl;

            csv << baseline << ","
                << res.epsilon << ","
                << threads << ","
                << r << ","
                << res.avg_latency_ns << ","
                << (res.wall_ns / 1e9) << ","
                << res.hit_ratio << ","
                << res.logical_ios << ","
                << res.physical_ios << ","
                << res.data_io_ns << "\n";
            csv.flush();
        }
    }

    csv.close();
    std::cout << "Results written to: " << output_csv << "\n";
    return 0;
}
