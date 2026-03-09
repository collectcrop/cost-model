#define _POSIX_C_SOURCE 200809L
#include <CLI/CLI.hpp>
#include "bplustree/stx_disk_kv.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <vector>
#include <string>
#include <cmath>
#include <errno.h>
#include <memory>
#include <algorithm>
#include <iostream>
#include <limits>

#include "FALCON/utils/include.hpp"
#include "FALCON/utils/utils.hpp"

static const size_t REC_LEN      = 8;    
static const bool   USE_O_DIRECT = true; 

#define MAX_THREADS (1 << 14)
#define RUNS 3

static inline uint64_t ns_between(const timespec &a, const timespec &b) {
  return (uint64_t)(b.tv_sec - a.tv_sec) * 1000000000ull
       + (uint64_t)(b.tv_nsec - a.tv_nsec);
}

static inline double wall_between(const struct timespec& a, const struct timespec& b) {
  return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

static void summarize_latency(const uint64_t* lat_ns, uint64_t N,
                              double* avg_ns, uint64_t* p50, uint64_t* p90, uint64_t* p99)
{
  __uint128_t sum = 0;
  for (uint64_t i = 0; i < N; ++i) sum += lat_ns[i];
  *avg_ns = (double)sum / (double)N;

  std::vector<uint64_t> v(lat_ns, lat_ns + N);
  auto nth = [&](double q) {
    uint64_t k = (uint64_t)std::min<double>(N-1, std::floor(q * (N-1)));
    std::nth_element(v.begin(), v.begin()+k, v.end());
    return v[k];
  };
  *p50 = nth(0.50);
  *p90 = nth(0.90);
  *p99 = nth(0.99);
}

struct thread_args_range {
  StxDiskKV*         idx;
  const falcon::RangeQ* ranges;
  uint64_t           start;
  uint64_t           end;
  size_t             rec_len;
  uint64_t*          lat_ns;       
  uint64_t           result_keys;  
};

static uint64_t do_range_query(StxDiskKV* idx, uint64_t lo, uint64_t hi, size_t rec_len) {
  if (lo > hi) std::swap(lo, hi);

  struct RangeInfo {
    uint64_t first_off = std::numeric_limits<uint64_t>::max();
    uint64_t last_off  = 0;
    uint64_t cnt       = 0;
  } info;

  auto cb = [&](StxDiskKV::Key k, StxDiskKV::Offset off) {
    (void)k;
    if (info.cnt == 0) {
      info.first_off = off;
      info.last_off  = off;
    } else {
      info.last_off  = off;       
    }
    ++info.cnt;
  };

  size_t scanned = idx->range_scan(lo, hi, cb);
  if (scanned == 0 || info.cnt == 0) return 0;


  uint64_t span_bytes = (info.last_off - info.first_off) + rec_len;

  std::vector<char> buf(span_bytes);
  ssize_t n = idx->read_at(info.first_off, buf.data(), (size_t)span_bytes);
  if (n < 0) {
    return 0;
  }

  return info.cnt;
}

void* range_worker(void* arg) {
  thread_args_range* args = (thread_args_range*)arg;
  uint64_t local_results = 0;

  for (uint64_t i = args->start; i < args->end; ++i) {
    const auto& rq = args->ranges[i];
    uint64_t lo = rq.lo;
    uint64_t hi = rq.hi;
    if (lo > hi) std::swap(lo, hi);

    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    uint64_t cnt = do_range_query(args->idx, lo, hi, args->rec_len);

    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);
    args->lat_ns[i] = ns_between(t0, t1);

    local_results += cnt;
  }

  args->result_keys = local_results;
  return nullptr;
}

static int build_index_from_data(StxDiskKV& kv, const char* data_path, size_t rec_len, uint64_t nkeys) {
  int rc = kv.open_data_file(data_path, USE_O_DIRECT);
  if (rc < 0) {
    fprintf(stderr, "open_data_file failed: %s rc=%d\n", data_path, rc);
    return -1;
  }

  std::vector<KeyType> keys = load_data_pgm_safe<KeyType>(data_path, nkeys);

  std::vector<uint64_t> offs(nkeys);
  for (uint64_t i = 0; i < nkeys; ++i) offs[i] = (uint64_t)i * (uint64_t)rec_len;

  kv.bulk_build(keys, offs.data(), (size_t)nkeys);

  return 0;
}

static int run_ranges_multithread(StxDiskKV& kv,
                                  const falcon::RangeQ* ranges,
                                  uint64_t total_ranges,
                                  int threads,
                                  size_t rec_len,
                                  double* out_latency_ns,
                                  double* out_time_s,
                                  double* out_iops,
                                  uint64_t* out_p50, uint64_t* out_p90, uint64_t* out_p99,
                                  double* out_avg_results)
{
  if (threads <= 0 || threads > MAX_THREADS) return -1;

  std::unique_ptr<uint64_t[]> lat_ns(new uint64_t[total_ranges]);

  pthread_t* tids = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)threads);
  thread_args_range* args = (thread_args_range*)malloc(sizeof(thread_args_range) * (size_t)threads);
  if (!tids || !args) { free(tids); free(args); return -1; }

  uint64_t per_thread = total_ranges / (uint64_t)threads;
  struct timespec ts1, ts2;
  clock_gettime(CLOCK_MONOTONIC, &ts1);

  for (int i = 0; i < threads; i++) {
    args[i].idx      = &kv;
    args[i].ranges   = ranges;
    args[i].start    = (uint64_t)i * per_thread;
    args[i].end      = (i == threads - 1) ? total_ranges : ((uint64_t)(i + 1) * per_thread);
    args[i].rec_len  = rec_len;
    args[i].lat_ns   = lat_ns.get();
    args[i].result_keys = 0;
    int rc = pthread_create(&tids[i], nullptr, range_worker, &args[i]);
    if (rc != 0) {
      fprintf(stderr, "pthread_create failed at %d\n", i);
      threads = i; // 只 join 已创建的
      break;
    }
  }

  uint64_t total_results = 0;
  for (int i = 0; i < threads; i++) {
    pthread_join(tids[i], nullptr);
    total_results += args[i].result_keys;
  }

  clock_gettime(CLOCK_MONOTONIC, &ts2);
  double seconds   = wall_between(ts1, ts2);
  double iops      = (double)total_ranges / seconds;

  double avg_ns; uint64_t p50, p90, p99;
  summarize_latency(lat_ns.get(), total_ranges, &avg_ns, &p50, &p90, &p99);

  *out_latency_ns = avg_ns;
  *out_p50        = p50;
  *out_p90        = p90;
  *out_p99        = p99;
  *out_time_s     = seconds;
  *out_iops       = iops;
  *out_avg_results = total_ranges ? (double)total_results / (double)total_ranges : 0.0;

  free(args);
  free(tids);
  return 0;
}

// int main(int argc, char* argv[]) {
//   if (argc < 3) {
//     fprintf(stderr, "Usage: %s <dataset_basename> <num_keys> [max_log2_threads]\n", argv[0]);
//     fprintf(stderr, "Example: %s wiki_ts_200M_uint64_unique 200000000 10\n", argv[0]);
//     return 1;
//   }

//   std::string dataset = argv[1];
//   (void)argv[2];

//   int max_exp = 10;
//   if (argc >= 4) {
//     max_exp = atoi(argv[3]);
//     if (max_exp < 0) max_exp = 0;
//   }

//   std::string data_path  = std::string("/mnt/home/zwshi/Datasets/SOSD/") + dataset;
//   std::string range_path = data_path + ".range.bin";

//   std::vector<falcon::RangeQ> ranges_vec = load_ranges_pgm_safe(range_path);
//   if (ranges_vec.empty()) {
//     fprintf(stderr, "Failed to load ranges from %s\n", range_path.c_str());
//     return -1;
//   }
//   uint64_t total_ranges = (uint64_t)ranges_vec.size();
//   printf("Loaded %lu ranges from %s\n", (unsigned long)total_ranges, range_path.c_str());

//   StxDiskKV kv;
//   if (build_index_from_data(kv, data_path.c_str(), REC_LEN) != 0) {
//     return -1;
//   }
//   printf("Index built: %zu entries, record_len=%zu, direct_io=%s\n",
//          kv.size(), REC_LEN, USE_O_DIRECT ? "true" : "false");

//   FILE* csv = fopen((dataset + "bptree_range_multithread.csv").c_str(), "w");
//   if (!csv) { perror("open csv"); return -1; }
//   fprintf(csv, "threads,avg_latency_ns,avg_walltime_s,avg_iops,avg_results,p50_ns,p90_ns,p99_ns\n");

//   for (int t = 0; t <= max_exp; t++) {
//     int threads = 1 << t;
//     if (threads > (int)total_ranges) threads = (int)total_ranges;
//     if (threads <= 0) threads = 1;

//     double total_latency = 0.0, total_time = 0.0, total_iops = 0.0, total_avg_res = 0.0;
//     uint64_t p50=0, p90=0, p99=0;

//     for (int r = 0; r < RUNS; r++) {
//       double latency=0.0, seconds=0.0, iops=0.0, avg_res=0.0;
//       if (run_ranges_multithread(kv,
//                                  ranges_vec.data(),
//                                  total_ranges,
//                                  threads,
//                                  REC_LEN,
//                                  &latency, &seconds, &iops,
//                                  &p50, &p90, &p99,
//                                  &avg_res) != 0) {
//         fprintf(stderr, "Failed run (threads=%d, run=%d)\n", threads, r);
//         continue;
//       }
//       total_latency += latency;
//       total_time    += seconds;
//       total_iops    += iops;
//       total_avg_res += avg_res;

//       printf("[run=%d] threads=%d, time=%.6f s, avg_lat=%.2f ns, iops=%.2f, avg_results=%.2f\n",
//              r, threads, seconds, latency, iops, avg_res);
//     }

//     double avg_latency = total_latency / RUNS;
//     double avg_time    = total_time    / RUNS;
//     double avg_iops    = total_iops    / RUNS;
//     double avg_res     = total_avg_res / RUNS;

//     fprintf(csv, "%d,%.2f,%.6f,%.2f,%.2f,%lu,%lu,%lu\n",
//             threads,
//             avg_latency,
//             avg_time,
//             avg_iops,
//             avg_res,
//             (unsigned long)p50,
//             (unsigned long)p90,
//             (unsigned long)p99);
//     fflush(csv);

//     printf("Threads=%d DONE: avg_lat=%.2f ns, avg_time=%.6f s, avg_iops=%.2f, avg_results=%.2f\n",
//            threads, avg_latency, avg_time, avg_iops, avg_res);
//   }

//   fclose(csv);
//   return 0;
// }

int main(int argc, char* argv[]) {
    CLI::App app{"B+Tree range multithread benchmark"};

    std::string dataset_basename;
    uint64_t num_keys = 0;
    int max_exp = 10;
    int repeat = RUNS;
    std::string baseline = "B+Tree";
    std::string output_csv;

    app.add_option(
        "--dataset",
        dataset_basename,
        "Dataset basename, e.g. wiki_ts_200M_uint64_unique"
    )->required();

    app.add_option(
        "--keys",
        num_keys,
        "Number of keys in the dataset"
    )->required();

    app.add_option(
        "--max-exp",
        max_exp,
        "Maximum thread exponent; test 1,2,4,...,2^max-exp"
    )->default_val(10);

    app.add_option(
        "--repeat",
        repeat,
        "Number of repetitions for each thread count"
    )->default_val(RUNS);

    app.add_option(
        "--baseline",
        baseline,
        "Baseline name written to CSV"
    )->default_val("B+Tree");

    app.add_option(
        "--output",
        output_csv,
        "Output CSV filename (default: <dataset>_bptree_range_multithread.csv)"
    );

    CLI11_PARSE(app, argc, argv);

    if (max_exp < 0) max_exp = 0;
    if (repeat <= 0) repeat = 1;

    std::string data_path  = falcon::DATASETS + dataset_basename;
    std::string range_path = data_path + ".range.bin";

    if (output_csv.empty()) {
        output_csv = dataset_basename + "_bptree_range_multithread.csv";
    }

    std::cout << "Loading ranges from: " << range_path << "\n";
    std::vector<falcon::RangeQ> ranges_vec = load_ranges_pgm_safe(range_path);
    if (ranges_vec.empty()) {
        std::cerr << "Failed to load ranges from " << range_path << "\n";
        return 1;
    }

    uint64_t total_ranges = static_cast<uint64_t>(ranges_vec.size());
    std::cout << "Loaded " << total_ranges << " ranges from " << range_path << "\n";

    std::cout << "Building B+Tree index from: " << data_path
              << " with " << num_keys << " keys\n";

    StxDiskKV kv;
    if (build_index_from_data(kv, data_path.c_str(), REC_LEN, num_keys) != 0) {
        std::cerr << "Failed to build B+Tree index from data file: " << data_path << "\n";
        return 1;
    }

    std::printf("Index built: %zu entries, record_len=%zu, direct_io=%s\n",
                kv.size(), REC_LEN, USE_O_DIRECT ? "true" : "false");

    FILE* csv = std::fopen(output_csv.c_str(), "w");
    if (!csv) {
        std::perror("open csv");
        return 1;
    }

    std::fprintf(csv,
                 "baseline,threads,latency_ns,walltime_s,avg_iops,avg_results,p50_ns,p90_ns,p99_ns\n");

    for (int t = 0; t <= max_exp; t++) {
        int threads = 1 << t;
        if (threads > static_cast<int>(total_ranges)) {
            threads = static_cast<int>(total_ranges == 0 ? 1 : total_ranges);
        }
        if (threads <= 0) threads = 1;

        std::cout << "Testing threads=" << threads
                  << ", repeats=" << repeat << " ...\n";

        double total_latency = 0.0;
        double total_time = 0.0;
        double total_iops = 0.0;
        double total_avg_res = 0.0;

        uint64_t p50 = 0, p90 = 0, p99 = 0;

        for (int r = 0; r < repeat; r++) {
            double latency = 0.0;
            double seconds = 0.0;
            double iops = 0.0;
            double avg_res = 0.0;
            uint64_t run_p50 = 0, run_p90 = 0, run_p99 = 0;

            if (run_ranges_multithread(kv,
                                       ranges_vec.data(),
                                       total_ranges,
                                       threads,
                                       REC_LEN,
                                       &latency,
                                       &seconds,
                                       &iops,
                                       &run_p50,
                                       &run_p90,
                                       &run_p99,
                                       &avg_res) != 0) {
                std::fprintf(stderr, "Failed run (threads=%d, run=%d)\n", threads, r);
                continue;
            }

            total_latency += latency;
            total_time    += seconds;
            total_iops    += iops;
            total_avg_res += avg_res;

            p50 = run_p50;
            p90 = run_p90;
            p99 = run_p99;

            std::printf("[%s][run=%d/%d][T=%d] time=%.6f s, avg_lat=%.2f ns, iops=%.2f, avg_results=%.2f, "
                        "p50=%llu, p90=%llu, p99=%llu\n",
                        baseline.c_str(),
                        r + 1, repeat,
                        threads,
                        seconds,
                        latency,
                        iops,
                        avg_res,
                        static_cast<unsigned long long>(run_p50),
                        static_cast<unsigned long long>(run_p90),
                        static_cast<unsigned long long>(run_p99));
        }

        double avg_latency = total_latency / repeat;
        double avg_time    = total_time    / repeat;
        double avg_iops    = total_iops    / repeat;
        double avg_res     = total_avg_res / repeat;

        std::fprintf(csv,
                     "%s,%d,%.2f,%.6f,%.2f,%.2f,%llu,%llu,%llu\n",
                     baseline.c_str(),
                     threads,
                     avg_latency,
                     avg_time,
                     avg_iops,
                     avg_res,
                     static_cast<unsigned long long>(p50),
                     static_cast<unsigned long long>(p90),
                     static_cast<unsigned long long>(p99));
        std::fflush(csv);

        std::printf("Threads=%d DONE: avg_lat=%.2f ns, avg_time=%.6f s, avg_iops=%.2f, avg_results=%.2f\n",
                    threads, avg_latency, avg_time, avg_iops, avg_res);
    }

    std::fclose(csv);
    std::cout << "Finished. Results saved to " << output_csv << "\n";
    return 0;
}
