#define _POSIX_C_SOURCE 200809L
#include "bplustree.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <math.h>

#define MAX_THREADS (1 << 14)
#define RUNS 3

struct thread_args {
    struct bplus_tree* tree;
    uint64_t* queries;
    uint64_t start;
    uint64_t end;
    uint64_t found;
};

void* query_worker(void* arg) {
    struct thread_args* args = (struct thread_args*)arg;
    uint64_t found = 0;

    for (uint64_t i = args->start; i < args->end; i++) {
        long result = bplus_tree_get(args->tree, (btree_key_t)args->queries[i]);
        if (result != -1) {
            found++;
        }
    }
    args->found = found;
    return NULL;
}

int run_queries_multithread(char* index_path, uint64_t* queries, uint64_t total_queries, int threads, double* out_latency_ns, double* out_time_s, double* out_iops) {
    pthread_t* tids = malloc(sizeof(pthread_t) * threads);
    struct thread_args* args = malloc(sizeof(struct thread_args) * threads);
    // struct bplus_tree** trees = malloc(sizeof(struct bplus_tree*) * threads);

    uint64_t per_thread = total_queries / threads;

    struct bplus_tree* tree = bplus_tree_init(index_path, 4096);
    // 初始化每线程 B+tree 实例
    // for (int i = 0; i < threads; i++) {
    //     trees[i] = tree;
    //     // printf("height %d \n", trees[i]->level);
    //     if (!trees[i]) {
    //         fprintf(stderr, "Thread %d failed to init tree\n", i);
    //         return -1;
    //     }
    // }

    struct timespec ts1, ts2;
    clock_gettime(CLOCK_MONOTONIC, &ts1);

    // 启动线程
    for (int i = 0; i < threads; i++) {
        args[i].tree = tree;
        args[i].queries = queries;
        args[i].start = i * per_thread;
        args[i].end = (i == threads - 1) ? total_queries : (i + 1) * per_thread;
        args[i].found = 0;
        pthread_create(&tids[i], NULL, query_worker, &args[i]);
    }

    uint64_t found_total = 0;
    for (int i = 0; i < threads; i++) {
        pthread_join(tids[i], NULL);
        found_total += args[i].found;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts2);
    double seconds = (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec) / 1e9;
    double latency_ns = (seconds * 1e9) / total_queries;
    double iops = total_queries / seconds;

    *out_latency_ns = latency_ns;
    *out_time_s = seconds;
    *out_iops = iops;

    // 清理
    // for (int i = 0; i < threads; i++) {
    //     bplus_tree_deinit(trees[i]);
    // }
    // free(trees);
    free(args);
    free(tids);

    return 0;
}

void build_tree(char* index_path, char* data_path) {
    struct bplus_tree* tree = bplus_tree_init(index_path, 4096);
    if (!tree) {
        fprintf(stderr, "Failed to initialize B+ tree\n");
        return;
    }

    FILE* data_file = fopen(data_path, "rb");
    if (!data_file) {
        perror("Failed to open data file");
        bplus_tree_deinit(tree);
        return;
    }

    fseek(data_file, 0, SEEK_END);
    long file_size = ftell(data_file);
    fseek(data_file, 0, SEEK_SET);

    uint64_t total_keys = file_size / sizeof(uint64_t);
    uint64_t* keys = malloc(file_size);
    if (!keys) {
        fprintf(stderr, "Failed to allocate memory for keys\n");
        fclose(data_file);
        bplus_tree_deinit(tree);
        return;
    }

    if (fread(keys, sizeof(uint64_t), total_keys, data_file) != total_keys) {
        fprintf(stderr, "Failed to read all keys\n");
        free(keys);
        fclose(data_file);
        bplus_tree_deinit(tree);
        return;
    }
    fclose(data_file);

    // 插入所有键值对，值设为 key + 1
    for (uint64_t i = 0; i < total_keys; i++) {
        if (bplus_tree_put(tree, (btree_key_t)keys[i], (btree_key_t)(keys[i] + 1)) != 0) {
            fprintf(stderr, "Failed to insert key %lu\n", keys[i]);
        }
    }

    printf("Inserted %lu keys into B+ tree\n", total_keys);

    free(keys);
    bplus_tree_deinit(tree);
}


int main(int argc, char* argv[]) {
    char* query_path   = "/mnt/home/zwshi/Datasets/SOSD/books_10M_uint64_unique.query.bin";
    char* data_path = "/mnt/home/zwshi/Datasets/SOSD/books_10M_uint64_unique";
    char* index_path   = "./index";
    build_tree(index_path, data_path);
    // 读取 query
    FILE* query_file = fopen(query_path, "rb");
    if (!query_file) {
        perror("Failed to open query file");
        return -1;
    }
    fseek(query_file, 0, SEEK_END);
    long file_size = ftell(query_file);
    fseek(query_file, 0, SEEK_SET);

    uint64_t total_queries = file_size / sizeof(uint64_t);
    uint64_t* queries = malloc(file_size);
    if (!queries) {
        fprintf(stderr, "Failed to alloc queries\n");
        fclose(query_file);
        return -1;
    }
    fread(queries, sizeof(uint64_t), total_queries, query_file);
    fclose(query_file);

    FILE* csv = fopen("bptree_multithread.csv", "w");
    fprintf(csv, "threads,avg_latency_ns,avg_walltime_s,height,avg_IOs\n");

    for (int t = 0; t <= 14; t++) {
        int threads = 1 << t;
        double total_latency = 0, total_time = 0, total_iops = 0;

        for (int r = 0; r < RUNS; r++) {
            double latency, seconds, iops;
            if (run_queries_multithread(index_path, queries, total_queries, threads, &latency, &seconds, &iops) != 0) {
                fprintf(stderr, "Failed run (threads=%d, run=%d)\n", threads, r);
                continue;
            }
            total_latency += latency;
            total_time += seconds;
            total_iops += iops;
            printf("thread=%d, time=%.6f s, latency=%.2f ns, iops=%.2f\n", threads, seconds, latency, iops);
        }

        fprintf(csv, "%d,%.2f,%.6f,%d,%.2f\n",
                threads,
                total_latency / RUNS,
                total_time / RUNS,
                0, // height: bplus_tree 没有直接返回高度，就填 0
                total_iops / RUNS);
        fflush(csv);

        printf("Threads=%d done\n", threads);
    }

    fclose(csv);
    free(queries);

    return 0;
}