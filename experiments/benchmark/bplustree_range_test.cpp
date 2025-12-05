#define _POSIX_C_SOURCE 200809L
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

#include "utils/include.hpp"
#include "utils/utils.hpp"

// ===== 参数按需调整 =====
static const size_t REC_LEN      = 8;    // SOSD key-only 文件：8 字节/条
static const bool   USE_O_DIRECT = true; // 是否使用 O_DIRECT

#define MAX_THREADS (1 << 14)
#define RUNS 3

static inline uint64_t ns_between(const timespec &a, const timespec &b) {
  return (uint64_t)(b.tv_sec - a.tv_sec) * 1000000000ull
       + (uint64_t)(b.tv_nsec - a.tv_nsec);
}

static inline double wall_between(const struct timespec& a, const struct timespec& b) {
  return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

// 分位数统计
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
  const pgm::RangeQ* ranges;
  uint64_t           start;
  uint64_t           end;
  size_t             rec_len;
  uint64_t*          lat_ns;       // 全局 latency 数组
  uint64_t           result_keys;  // 本线程总命中 key 数
};

// ====== 关键：每个 range 一次性 IO ======
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
      info.last_off  = off;        // 索引是按 key 升序的，offset 也递增
    }
    ++info.cnt;
  };

  size_t scanned = idx->range_scan(lo, hi, cb);
  if (scanned == 0 || info.cnt == 0) return 0;

  // 这段区间在数据文件上是连续的：从 first_off 到 last_off。
  uint64_t span_bytes = (info.last_off - info.first_off) + rec_len;

  std::vector<char> buf(span_bytes);
  ssize_t n = idx->read_at(info.first_off, buf.data(), (size_t)span_bytes);
  if (n < 0) {
    // IO 出错时简单返回 0
    return 0;
  }

  // 这里只是触发 IO，真实扫描在 range_scan 里已经做了
  return info.cnt;
}

// 线程函数：对 [start, end) 的 ranges 做查询
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

// 读取二进制 uint64_t（用于建索引的 SOSD 数据文件）
static uint64_t* load_bin64(const char* path, uint64_t& out_count) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror("fopen(bin64)"); return nullptr; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz <= 0 || (sz % (long)sizeof(uint64_t)) != 0) {
    fprintf(stderr, "bad bin64 file size: %ld\n", sz);
    fclose(f);
    return nullptr;
  }
  out_count = (uint64_t)(sz / (long)sizeof(uint64_t));
  uint64_t* a = (uint64_t*)malloc((size_t)sz);
  if (!a) { fclose(f); fprintf(stderr, "malloc failed\n"); return nullptr; }
  size_t nread = fread(a, sizeof(uint64_t), (size_t)out_count, f);
  fclose(f);
  if (nread != out_count) {
    fprintf(stderr, "fread mismatch: %zu != %lu\n", nread, out_count);
    free(a);
    return nullptr;
  }
  return a;
}

// 基于“固定长度记录”的索引构建：offset = i * REC_LEN
static int build_index_from_data(StxDiskKV& kv, const char* data_path, size_t rec_len) {
  int rc = kv.open_data_file(data_path, USE_O_DIRECT);
  if (rc < 0) {
    fprintf(stderr, "open_data_file failed: %s rc=%d\n", data_path, rc);
    return -1;
  }

  uint64_t nkeys = 0;
  uint64_t* keys = load_bin64(data_path, nkeys);
  if (!keys) return -1;

  std::vector<uint64_t> offs(nkeys);
  for (uint64_t i = 0; i < nkeys; ++i) offs[i] = (uint64_t)i * (uint64_t)rec_len;

  kv.bulk_build(keys, offs.data(), (size_t)nkeys);

  free((void*)keys);
  return 0;
}

// 多线程跑 range workload
static int run_ranges_multithread(StxDiskKV& kv,
                                  const pgm::RangeQ* ranges,
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

int main(int argc, char* argv[]) {
  // Usage:
  //   ./bptree_range_multithread <dataset_basename> <num_keys> [max_log2_threads]
  //
  // Example:
  //   ./bptree_range_multithread wiki_ts_200M_uint64_unique 200000000 10
  //
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <dataset_basename> <num_keys> [max_log2_threads]\n", argv[0]);
    fprintf(stderr, "Example: %s wiki_ts_200M_uint64_unique 200000000 10\n", argv[0]);
    return 1;
  }

  std::string dataset = argv[1];
  (void)argv[2]; // num_keys 当前没用，可以后续做 sanity check

  int max_exp = 10;
  if (argc >= 4) {
    max_exp = atoi(argv[3]);
    if (max_exp < 0) max_exp = 0;
  }

  std::string data_path  = std::string("/mnt/home/zwshi/Datasets/SOSD/") + dataset;
  std::string range_path = data_path + ".range.bin";

  // 1) 读取 range workload
  std::vector<pgm::RangeQ> ranges_vec = load_ranges_pgm_safe(range_path);
  if (ranges_vec.empty()) {
    fprintf(stderr, "Failed to load ranges from %s\n", range_path.c_str());
    return -1;
  }
  uint64_t total_ranges = (uint64_t)ranges_vec.size();
  printf("Loaded %lu ranges from %s\n", (unsigned long)total_ranges, range_path.c_str());

  // 2) 构建索引 + 打开数据文件
  StxDiskKV kv;
  if (build_index_from_data(kv, data_path.c_str(), REC_LEN) != 0) {
    return -1;
  }
  printf("Index built: %zu entries, record_len=%zu, direct_io=%s\n",
         kv.size(), REC_LEN, USE_O_DIRECT ? "true" : "false");

  // 3) 多线程 range benchmark
  FILE* csv = fopen((dataset + "bptree_range_multithread.csv").c_str(), "w");
  if (!csv) { perror("open csv"); return -1; }
  fprintf(csv, "threads,avg_latency_ns,avg_walltime_s,avg_iops,avg_results,p50_ns,p90_ns,p99_ns\n");

  for (int t = 0; t <= max_exp; t++) {
    int threads = 1 << t;
    if (threads > (int)total_ranges) threads = (int)total_ranges;
    if (threads <= 0) threads = 1;

    double total_latency = 0.0, total_time = 0.0, total_iops = 0.0, total_avg_res = 0.0;
    uint64_t p50=0, p90=0, p99=0;

    for (int r = 0; r < RUNS; r++) {
      double latency=0.0, seconds=0.0, iops=0.0, avg_res=0.0;
      if (run_ranges_multithread(kv,
                                 ranges_vec.data(),
                                 total_ranges,
                                 threads,
                                 REC_LEN,
                                 &latency, &seconds, &iops,
                                 &p50, &p90, &p99,
                                 &avg_res) != 0) {
        fprintf(stderr, "Failed run (threads=%d, run=%d)\n", threads, r);
        continue;
      }
      total_latency += latency;
      total_time    += seconds;
      total_iops    += iops;
      total_avg_res += avg_res;

      printf("[run=%d] threads=%d, time=%.6f s, avg_lat=%.2f ns, iops=%.2f, avg_results=%.2f\n",
             r, threads, seconds, latency, iops, avg_res);
    }

    double avg_latency = total_latency / RUNS;
    double avg_time    = total_time    / RUNS;
    double avg_iops    = total_iops    / RUNS;
    double avg_res     = total_avg_res / RUNS;

    fprintf(csv, "%d,%.2f,%.6f,%.2f,%.2f,%lu,%lu,%lu\n",
            threads,
            avg_latency,
            avg_time,
            avg_iops,
            avg_res,
            (unsigned long)p50,
            (unsigned long)p90,
            (unsigned long)p99);
    fflush(csv);

    printf("Threads=%d DONE: avg_lat=%.2f ns, avg_time=%.6f s, avg_iops=%.2f, avg_results=%.2f\n",
           threads, avg_latency, avg_time, avg_iops, avg_res);
  }

  fclose(csv);
  return 0;
}
