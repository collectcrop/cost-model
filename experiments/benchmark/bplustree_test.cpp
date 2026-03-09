#define _POSIX_C_SOURCE 200809L
#include <CLI/CLI.hpp>
#include "bplustree/stx_disk_kv.h"  
#include "FALCON/utils/include.hpp"
#include "FALCON/utils/utils.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <vector>
#include <string>
#include <cmath>
#include <errno.h>

static const bool   USE_O_DIRECT = true;  
static inline uint64_t ns_between(const timespec &a, const timespec &b) {
  return (uint64_t)(b.tv_sec - a.tv_sec) * 1000000000ull
       + (uint64_t)(b.tv_nsec - a.tv_nsec);
}

struct thread_args {
  StxDiskKV*            idx;      
  std::vector<KeyType>* queries;  
  uint64_t              start;   
  uint64_t              end;      
  uint64_t              found;    
  size_t                rec_len;  
  uint64_t*             lat_ns; 
  uint64_t              thread_cpu_ns;   
  uint64_t              thread_wall_ns;  
};

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
static inline double wall_between(const struct timespec& a, const struct timespec& b) {
  return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}
void* query_worker(void* arg) {
  thread_args* args = (thread_args*)arg;
  std::vector<char> buf(args->rec_len);
  uint64_t found = 0;

  timespec c0{}, c1{};
  timespec w0{}, w1{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c0);
  clock_gettime(CLOCK_MONOTONIC, &w0);

  for (uint64_t i = args->start; i < args->end; i++) {
    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    ssize_t n = args->idx->get_record((*args->queries)[i], buf.data(), args->rec_len);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    args->lat_ns[i] = ns_between(t0, t1);   
    if (n > 0) found++;
  }
  clock_gettime(CLOCK_MONOTONIC, &w1);
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c1);
  args->found = found;

  args->thread_wall_ns = ns_between(w0, w1);
  args->thread_cpu_ns  = ns_between(c0, c1);
  return nullptr;
}
static int build_index_from_data(StxDiskKV& kv, const char* data_path, size_t rec_len) {
  int rc = kv.open_data_file(data_path, USE_O_DIRECT);
  if (rc < 0) {
    fprintf(stderr, "open_data_file failed: %s rc=%d\n", data_path, rc);
    return -1;
  }

  std::vector<KeyType> keys = load_binary<KeyType>(data_path);
  size_t nkeys = keys.size();
  std::vector<uint64_t> offs(nkeys);
  for (size_t i = 0; i < nkeys; ++i) offs[i] = (uint64_t)i * (uint64_t)rec_len;
  kv.bulk_build(keys, offs.data(), nkeys);

  return 0;
}

static int run_queries_multithread(StxDiskKV& kv,
                                   std::vector<KeyType> &queries,
                                   KeyType total_queries,
                                   int threads,
                                   size_t rec_len,
                                   double* out_latency_ns,
                                   double* out_time_s,
                                   double* out_iops,
                                   KeyType* out_p50, KeyType* out_p90, KeyType* out_p99,
                                   double* out_mem_time_s, double* out_io_time_s) {
  if (threads <= 0) return -1;

  std::unique_ptr<KeyType[]> lat_ns(new KeyType[total_queries]);

  pthread_t* tids = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)threads);
  thread_args* args = (thread_args*)malloc(sizeof(thread_args) * (size_t)threads);
  if (!tids || !args) { free(tids); free(args); return -1; }

  KeyType per_thread = total_queries / (KeyType)threads;
  struct timespec ts1, ts2;
  clock_gettime(CLOCK_MONOTONIC, &ts1);

  for (int i = 0; i < threads; i++) {
    args[i].idx     = &kv;
    args[i].queries = &queries;
    args[i].start   = (KeyType)i * per_thread;
    args[i].end     = (i == threads - 1) ? total_queries : ((KeyType)(i + 1) * per_thread);
    args[i].found   = 0;
    args[i].rec_len = rec_len;
    args[i].lat_ns  = lat_ns.get();
    int rc = pthread_create(&tids[i], nullptr, query_worker, &args[i]);
    if (rc != 0) {
      fprintf(stderr, "pthread_create failed at %d\n", i);
      threads = i; 
      break;
    }
  }

  KeyType found_total = 0;
  KeyType total_thread_wall_ns = 0;
  KeyType total_thread_cpu_ns  = 0;
  for (int i = 0; i < threads; i++) {
    pthread_join(tids[i], nullptr);
    found_total += args[i].found;
    total_thread_wall_ns += args[i].thread_wall_ns;
    total_thread_cpu_ns  += args[i].thread_cpu_ns;
  }

  clock_gettime(CLOCK_MONOTONIC, &ts2);
  double seconds = wall_between(ts1, ts2);
  double latency_ns = (seconds * 1e9) / (double)total_queries;
  double iops = (double)total_queries / seconds;

  double avg_ns; uint64_t p50, p90, p99;
  summarize_latency(lat_ns.get(), total_queries, &avg_ns, &p50, &p90, &p99);
  *out_latency_ns = avg_ns; *out_p50 = p50; *out_p90 = p90; *out_p99 = p99;

  *out_time_s     = seconds;
  *out_iops       = iops;

  double mem_time_s = (double)total_thread_cpu_ns / 1e9;
  double io_ns      = 0;
  if (total_thread_wall_ns > total_thread_cpu_ns)
      io_ns = (double)(total_thread_wall_ns - total_thread_cpu_ns);
  double io_time_s  = io_ns / 1e9;
  if (out_mem_time_s) *out_mem_time_s = mem_time_s;
  if (out_io_time_s)  *out_io_time_s  = io_time_s;

  free(args);
  free(tids);
  return 0;
}

// int main(int argc, char* argv[]) {
//     // Usage
//     if (argc < 3) {
//         std::cerr << "Usage:\n  " << argv[0]
//                   << " <dataset_basename> <num_keys> [max_log2_threads] [repeats]\n"
//                   << "Example:\n  " << argv[0]
//                   << " wiki_ts_200M_uint64_unique 200000000 10 3\n";
//         return 1;
//     }

//     std::string dataset_basename = argv[1]; 
//     uint64_t num_keys = std::strtoull(argv[2], nullptr, 10);

//     int max_exp = 10; 
//     if (argc >= 4) {
//         max_exp = std::atoi(argv[3]);
//         if (max_exp < 0) max_exp = 0;
//     }

//     int repeats = 3;
//     if (argc >= 5) {
//         repeats = std::atoi(argv[4]);
//         if (repeats <= 0) repeats = 1;
//     }

//     std::string data_file  = std::string(falcon::DATASETS) + dataset_basename;
//     std::string query_file = std::string(falcon::DATASETS) + dataset_basename + ".query.bin";
  
//     std::vector<KeyType> queries = load_queries(query_file);
//     uint64_t total_queries = queries.size();

//     StxDiskKV kv;
//     if (build_index_from_data(kv, data_file.c_str(), sizeof(KeyType)) != 0) {
//       return -1;
//     }
//     printf("Index built: %zu entries, record_len=%zu, direct_io=%s\n",
//           kv.size(), sizeof(KeyType), USE_O_DIRECT ? "true" : "false");

//     std::string csv_name = dataset_basename + "_bplustree_multithread.csv";
//     FILE* csv = fopen(csv_name.c_str(), "w");
//     if (!csv) { perror("open csv"); return -1; }
//     fprintf(csv, "threads,avg_latency_ns,total_wall_time_s,avg_iops\n");

//     for (int t = 0; t <= max_exp; t++) { 
//       int threads = 1 << t;
//       uint64_t p50=0, p90=0, p99=0;
//       for (int r = 0; r < repeats; r++) {
//         double latency=0, seconds=0, iops=0;
//         double mem_s=0, io_s=0;
//         if (run_queries_multithread(kv, queries, total_queries, threads, sizeof(KeyType),
//                                     &latency, &seconds, &iops, &p50, &p90, &p99, 
//                                     &mem_s, &io_s) != 0) {
//           fprintf(stderr, "Failed run (threads=%d, run=%d)\n", threads, r);
//           continue;
//         }
//         fprintf(csv, "%d,%.2f,%.6f,%.2f\n",
//               threads,
//               latency,
//               seconds,
//               iops);
//         fflush(csv);
//         printf("thread=%d, time=%.6f s, latency=%.2f ns, iops=%.2f\n",
//               threads, seconds, latency, iops);
//       }
//       printf("Threads=%d done\n", threads);
//     }

//     fclose(csv);
//     return 0;
// }

int main(int argc, char* argv[]) {
    CLI::App app{"B+Tree multithread benchmark"};

    std::string dataset_basename;
    uint64_t num_keys = 0;
    int max_exp = 10;
    int repeat = 3;
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
    )->default_val(3);

    app.add_option(
        "--baseline",
        baseline,
        "Baseline name written to CSV"
    )->default_val("B+Tree");

    app.add_option(
        "--output",
        output_csv,
        "Output CSV filename (default: <dataset>_bplustree_multithread.csv)"
    );

    CLI11_PARSE(app, argc, argv);

    if (max_exp < 0) max_exp = 0;
    if (repeat <= 0) repeat = 1;

    std::string data_file  = std::string(falcon::DATASETS) + dataset_basename;
    std::string query_file = std::string(falcon::DATASETS) + dataset_basename + ".query.bin";

    if (output_csv.empty()) {
        output_csv = dataset_basename + "_bplustree_multithread.csv";
    }

    std::cout << "Loading queries from: " << query_file << "\n";
    std::vector<KeyType> queries = load_queries(query_file);
    uint64_t total_queries = queries.size();

    if (queries.empty()) {
        std::cerr << "Failed to load queries or query file is empty: " << query_file << "\n";
        return 1;
    }

    std::cout << "Building B+Tree index from: " << data_file
              << " with " << num_keys << " keys\n";

    StxDiskKV kv;
    if (build_index_from_data(kv, data_file.c_str(), sizeof(KeyType)) != 0) {
        std::cerr << "Failed to build B+Tree index from data file: " << data_file << "\n";
        return 1;
    }

    std::printf("Index built: %zu entries, record_len=%zu, direct_io=%s\n",
                kv.size(), sizeof(KeyType), USE_O_DIRECT ? "true" : "false");

    FILE* csv = std::fopen(output_csv.c_str(), "w");
    if (!csv) {
        std::perror("open csv");
        return 1;
    }

    std::fprintf(csv,
                 "baseline,threads,run_id,avg_latency_ns,total_wall_time_s,avg_iops,p50_ns,p90_ns,p99_ns,mem_time_s,io_time_s\n");

    for (int t = 0; t <= max_exp; t++) {
        int threads = 1 << t;
        if (threads <= 0) threads = 1;
        if (static_cast<uint64_t>(threads) > total_queries) {
            threads = static_cast<int>(total_queries == 0 ? 1 : total_queries);
        }

        std::cout << "Testing threads=" << threads
                  << ", repeats=" << repeat << " ...\n";

        for (int r = 0; r < repeat; r++) {
            double latency = 0.0, seconds = 0.0, iops = 0.0;
            double mem_s = 0.0, io_s = 0.0;
            uint64_t p50 = 0, p90 = 0, p99 = 0;

            if (run_queries_multithread(kv, queries, total_queries, threads, sizeof(KeyType),
                                        &latency, &seconds, &iops,
                                        &p50, &p90, &p99,
                                        &mem_s, &io_s) != 0) {
                std::fprintf(stderr, "Failed run (threads=%d, run=%d)\n", threads, r);
                continue;
            }

            std::fprintf(csv,
                         "%s,%d,%d,%.2f,%.6f,%.2f,%llu,%llu,%llu,%.6f,%.6f\n",
                         baseline.c_str(),
                         threads,
                         r,
                         latency,
                         seconds,
                         iops,
                         static_cast<unsigned long long>(p50),
                         static_cast<unsigned long long>(p90),
                         static_cast<unsigned long long>(p99),
                         mem_s,
                         io_s);
            std::fflush(csv);

            std::printf("[%s][run=%d/%d][T=%d] time=%.6f s, latency=%.2f ns, iops=%.2f, "
                        "p50=%llu, p90=%llu, p99=%llu, mem_s=%.6f, io_s=%.6f\n",
                        baseline.c_str(),
                        r + 1, repeat,
                        threads,
                        seconds,
                        latency,
                        iops,
                        static_cast<unsigned long long>(p50),
                        static_cast<unsigned long long>(p90),
                        static_cast<unsigned long long>(p99),
                        mem_s,
                        io_s);
        }

        std::printf("Threads=%d done\n", threads);
    }

    std::fclose(csv);
    std::cout << "Finished. Results saved to " << output_csv << "\n";
    return 0;
}