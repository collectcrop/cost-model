#define _POSIX_C_SOURCE 200809L
#include "stx_disk_kv.h"   // 前面我给你的封装：内存索引 + 数据文件 I/O
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <vector>
#include <string>
#include <cmath>
#include <errno.h>

// ===== 参数按需调整 =====
static const size_t REC_LEN = 8;          // SOSD key-only 数据文件每条记录长度（books_10M 为 8 字节）
static const bool   USE_O_DIRECT = true;  // 是否用 O_DIRECT 打开数据文件（建议 true，bounce buffer 已封装）

#define MAX_THREADS (1 << 14)
#define RUNS 3

static inline uint64_t ns_between(const timespec &a, const timespec &b) {
  return (uint64_t)(b.tv_sec - a.tv_sec) * 1000000000ull
       + (uint64_t)(b.tv_nsec - a.tv_nsec);
}

struct thread_args {
  StxDiskKV*     idx;      // 共享索引（线程安全）
  const uint64_t*queries;  // 查询 key 数组
  uint64_t       start;    // 起始下标（包含）
  uint64_t       end;      // 结束下标（不含）
  uint64_t       found;    // 线程命中数
  size_t         rec_len;  // 固定记录长度
  uint64_t*       lat_ns; 
};

static void summarize_latency(const uint64_t* lat_ns, uint64_t N,
                              double* avg_ns, uint64_t* p50, uint64_t* p90, uint64_t* p99)
{
  // 平均
  __uint128_t sum = 0;
  for (uint64_t i = 0; i < N; ++i) sum += lat_ns[i];
  *avg_ns = (double)sum / (double)N;

  // 分位数（不改变原数组：复制一份或做选择算法）
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

// 线程函数：存在性 + 拉取记录（如果只是存在性统计，可以改成 get_offset）
void* query_worker(void* arg) {
  thread_args* args = (thread_args*)arg;
  std::vector<char> buf(args->rec_len);
  uint64_t found = 0;

  for (uint64_t i = args->start; i < args->end; i++) {
    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);
    // 读取固定长度记录：>0 表示命中；对于 key-only 的 8 字节文件，n==8 即命中
    ssize_t n = args->idx->get_record(args->queries[i], buf.data(), args->rec_len);
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    args->lat_ns[i] = ns_between(t0, t1);   // 每条 query 的实际延迟（ns）
    if (n > 0) found++;
  }
  args->found = found;
  return nullptr;
}

// 读取二进制 uint64 文件（用于 queries 或数据构建时的 keys）
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
// data_path 指向 SOSD 主数据文件（例如 books_10M_uint64_unique），它通常是有序的 8 字节 key 列表
static int build_index_from_data(StxDiskKV& kv, const char* data_path, size_t rec_len) {
  // 用 O_DIRECT 打开数据文件（如果失败会返回负 errno）
  int rc = kv.open_data_file(data_path, USE_O_DIRECT);
  if (rc < 0) {
    fprintf(stderr, "open_data_file failed: %s rc=%d\n", data_path, rc);
    return -1;
  }

  // 读取所有 key（SOSD 数据文件本身就是一个 uint64_t 数组）
  uint64_t nkeys = 0;
  uint64_t* keys = load_bin64(data_path, nkeys);
  if (!keys) return -1;

  // 计算 offsets = i * rec_len
  std::vector<uint64_t> offs(nkeys);
  for (uint64_t i = 0; i < nkeys; ++i) offs[i] = (uint64_t)i * (uint64_t)rec_len;

  // 批量建索引（stx::btree 内存插入）
  kv.bulk_build(keys, offs.data(), (size_t)nkeys);

  free((void*)keys);
  return 0;
}

static int run_queries_multithread(StxDiskKV& kv,
                                   const uint64_t* queries,
                                   uint64_t total_queries,
                                   int threads,
                                   size_t rec_len,
                                   double* out_latency_ns,
                                   double* out_time_s,
                                   double* out_iops,
                                   uint64_t* out_p50, uint64_t* out_p90, uint64_t* out_p99) {
  if (threads <= 0 || threads > MAX_THREADS) return -1;

  std::unique_ptr<uint64_t[]> lat_ns(new uint64_t[total_queries]);

  pthread_t* tids = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)threads);
  thread_args* args = (thread_args*)malloc(sizeof(thread_args) * (size_t)threads);
  if (!tids || !args) { free(tids); free(args); return -1; }

  uint64_t per_thread = total_queries / (uint64_t)threads;
  struct timespec ts1, ts2;
  clock_gettime(CLOCK_MONOTONIC, &ts1);

  for (int i = 0; i < threads; i++) {
    args[i].idx     = &kv;
    args[i].queries = queries;
    args[i].start   = (uint64_t)i * per_thread;
    args[i].end     = (i == threads - 1) ? total_queries : ((uint64_t)(i + 1) * per_thread);
    args[i].found   = 0;
    args[i].rec_len = rec_len;
    args[i].lat_ns  = lat_ns.get();
    int rc = pthread_create(&tids[i], nullptr, query_worker, &args[i]);
    if (rc != 0) {
      fprintf(stderr, "pthread_create failed at %d\n", i);
      threads = i; // 只 join 已创建的
      break;
    }
  }

  uint64_t found_total = 0;
  for (int i = 0; i < threads; i++) {
    pthread_join(tids[i], nullptr);
    found_total += args[i].found;
  }

  clock_gettime(CLOCK_MONOTONIC, &ts2);
  double seconds = wall_between(ts1, ts2);
  double latency_ns = (seconds * 1e9) / (double)total_queries;
  double iops = (double)total_queries / seconds;

  // *out_latency_ns = latency_ns;
  // 真实 per-query 的平均与分位数
  double avg_ns; uint64_t p50, p90, p99;
  summarize_latency(lat_ns.get(), total_queries, &avg_ns, &p50, &p90, &p99);
  *out_latency_ns = avg_ns; *out_p50 = p50; *out_p90 = p90; *out_p99 = p99;

  *out_time_s     = seconds;
  *out_iops       = iops;

  free(args);
  free(tids);
  return 0;
}

int main(int argc, char* argv[]) {
  // 路径按需修改
  const char* data_path  = "/mnt/home/zwshi/Datasets/SOSD/wiki_ts_10M_uint64_unique";              // 主数据文件（8B/条）
  const char* query_path = "/mnt/home/zwshi/Datasets/SOSD/wiki_ts_10M_uint64_unique.query.bin";    // 查询文件（8B/条）

  // 1) 读取 queries
  uint64_t total_queries = 0;
  uint64_t* queries = load_bin64(query_path, total_queries);
  if (!queries) return -1;

  // 2) 构建内存索引 + 打开数据文件
  StxDiskKV kv;
  if (build_index_from_data(kv, data_path, REC_LEN) != 0) {
    free(queries);
    return -1;
  }
  printf("Index built: %zu entries, record_len=%zu, direct_io=%s\n",
         kv.size(), REC_LEN, USE_O_DIRECT ? "true" : "false");

  // 3) 多线程查询压测
  FILE* csv = fopen("stx_disk_kv_multithread.csv", "w");
  if (!csv) { perror("open csv"); free(queries); return -1; }
  fprintf(csv, "threads,avg_latency_ns,avg_walltime_s,avg_iops\n");

  for (int t = 0; t <= 14; t++) { 
    int threads = 1 << t;
    double total_latency = 0, total_time = 0, total_iops = 0;
    uint64_t p50=0, p90=0, p99=0;
    for (int r = 0; r < RUNS; r++) {
      double latency=0, seconds=0, iops=0;
      if (run_queries_multithread(kv, queries, total_queries, threads, REC_LEN,
                                  &latency, &seconds, &iops, &p50, &p90, &p99) != 0) {
        fprintf(stderr, "Failed run (threads=%d, run=%d)\n", threads, r);
        continue;
      }
      total_latency += latency;
      total_time    += seconds;
      total_iops    += iops;
      printf("thread=%d, time=%.6f s, latency=%.2f ns, iops=%.2f\n",
             threads, seconds, latency, iops);
    }

    fprintf(csv, "%d,%.2f,%.6f,%.2f\n",
            threads,
            total_latency / RUNS,
            total_time / RUNS,
            total_iops / RUNS);
    fflush(csv);

    printf("Threads=%d done\n", threads);
  }

  fclose(csv);
  free(queries);
  return 0;
}
