// io_bench.cpp
#define _GNU_SOURCE
#include <libaio.h>
#include <liburing.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

#define FILE_PATH "testfile"
#define FILE_SIZE (4UL << 30)   // 4 GB

const int BlockSize = 4096;
typedef struct {
    int fd;
    int tid;
    int qdepth;
    int ops;
    char *mode;   // "pread" / "libaio" / "io_uring"
} thread_arg_t;

double now_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void *worker_pread(void *arg) {
    thread_arg_t *t = (thread_arg_t *)arg;
    char *buf = (char*)aligned_alloc(4096, BlockSize);
    off_t offset;
    for (int i = 0; i < t->ops; i++) {
        offset = ((off_t)rand() % (FILE_SIZE / BlockSize)) * BlockSize;
        pread(t->fd, buf, BlockSize, offset);
    }
    free(buf);
    return NULL;
}

void *worker_libaio(void *arg) {
    thread_arg_t *t = (thread_arg_t *)arg;
    io_context_t ctx = 0;
    if (io_setup(t->qdepth, &ctx) < 0) {
        perror("io_setup");
        return NULL;
    }

    struct iocb **cbs = (struct iocb**)malloc(sizeof(struct iocb *) * t->qdepth);
    struct io_event *events = (struct io_event*)calloc(t->qdepth, sizeof(struct io_event));
    char **bufs = malloc(sizeof(char *) * t->qdepth);

    for (int i = 0; i < t->qdepth; i++) {
        bufs[i] = (char*)aligned_alloc(4096, BlockSize);
        cbs[i] = (struct iocb*)calloc(1, sizeof(struct iocb));
    }

    int submitted = 0, completed = 0;
    while (completed < t->ops) {
        // 提交新请求
        while (submitted - completed < t->qdepth && submitted < t->ops) {
            off_t offset = ((off_t)rand() % (FILE_SIZE / BlockSize)) * BlockSize;
            io_prep_pread(cbs[submitted % t->qdepth], t->fd, bufs[submitted % t->qdepth],
                          BlockSize, offset);
            cbs[submitted % t->qdepth]->data = bufs[submitted % t->qdepth];

            struct iocb *list[1] = { cbs[submitted % t->qdepth] };
            int ret;
            do {
                ret = io_submit(ctx, 1, list);
            } while (ret < 0 && errno == EAGAIN); // 避免提交失败直接丢

            if (ret == 1)
                submitted++;
        }

        // 回收完成事件
        int got = io_getevents(ctx, 0, t->qdepth, events, NULL);
        completed += got;
    }

    for (int i = 0; i < t->qdepth; i++) {
        free(bufs[i]);
        free(cbs[i]);
    }
    free(bufs);
    free(cbs);
    free(events);
    io_destroy(ctx);
    return NULL;
}

void *worker_io_uring(void *arg) {
    thread_arg_t *t = (thread_arg_t *)arg;
    struct io_uring ring;
    io_uring_queue_init(t->qdepth, &ring, 0);

    char **bufs = (char**)malloc(sizeof(char *) * t->qdepth);
    for (int i = 0; i < t->qdepth; i++)
        bufs[i] = (char*)aligned_alloc(4096, BlockSize);

    int submitted = 0, completed = 0;
    while (completed < t->ops) {
        while (submitted - completed < t->qdepth && submitted < t->ops) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            off_t offset = ((off_t)rand() % (FILE_SIZE / BlockSize)) * BlockSize;
            io_uring_prep_read(sqe, t->fd, bufs[submitted % t->qdepth], BlockSize, offset);
            submitted++;
        }
        io_uring_submit(&ring);

        struct io_uring_cqe *cqe;
        int ret = io_uring_wait_cqe(&ring, &cqe);
        if (ret == 0) {
            io_uring_cqe_seen(&ring, cqe);
            completed++;
        }
    }

    for (int i = 0; i < t->qdepth; i++) free(bufs[i]);
    free(bufs);
    io_uring_queue_exit(&ring);
    return NULL;
}

void *worker_io_uring_optimized(void *arg) {
    thread_arg_t *t = (thread_arg_t *)arg;
    struct io_uring ring;
    struct io_uring_params params;
    memset(&params, 0, sizeof(params));

    // Try to enable SQPOLL if available (not fatal if fails)
    params.flags = IORING_SETUP_SQPOLL;
    params.sq_thread_idle = 2000;

    // Make ring entries reasonably large to avoid full queue
    int entries = t->qdepth * 4;
    if (entries < 256) entries = 256;

    if (io_uring_queue_init_params(entries, &ring, &params) < 0) {
        // fallback: try without SQPOLL
        params.flags = 0;
        if (io_uring_queue_init(entries, &ring, 0) < 0) {
            perror("io_uring_queue_init fallback");
            return NULL;
        }
    }

    // allocate qdepth buffers (reuse)
    char **bufs = (char**)malloc(sizeof(char*) * t->qdepth);
    for (int i = 0; i < t->qdepth; ++i) {
        void *p = NULL;
        if (posix_memalign(&p, 4096, BlockSize) != 0) {
            perror("posix_memalign");
            // cleanup
            for (int j = 0; j < i; ++j) free(bufs[j]);
            free(bufs);
            io_uring_queue_exit(&ring);
            return NULL;
        }
        bufs[i] = (char*)p;
    }

    // Pre-generate offsets to avoid lock contention in rand()
    uint64_t total_ops = (uint64_t)t->ops;
    uint64_t *offsets = malloc(sizeof(uint64_t) * total_ops);
    unsigned int seed = (unsigned int)(time(NULL) ^ (uintptr_t)pthread_self() ^ (t->tid << 16));
    uint64_t max_block = FILE_SIZE / BlockSize;
    for (uint64_t i = 0; i < total_ops; ++i) {
        offsets[i] = (uint64_t)(rand_r(&seed) % max_block) * (uint64_t)BlockSize;
    }

    uint64_t submitted = 0, completed = 0;
    struct io_uring_cqe *cqe = NULL;

    while (completed < total_ops) {
        // submit until we reach per-thread qdepth outstanding or finish
        while (submitted < total_ops && (submitted - completed) < (uint64_t)t->qdepth) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            if (!sqe) break; // SQ full
            uint64_t idx = submitted % t->qdepth;
            off_t off = (off_t)offsets[submitted];
            io_uring_prep_read(sqe, t->fd, bufs[idx], BlockSize, off);
            // optionally store submitted id
            io_uring_sqe_set_data(sqe, (void*)(uintptr_t)submitted);
            submitted++;
        }

        // batch submit (if any)
        int ret = io_uring_submit(&ring);
        if (ret < 0 && ret != -EAGAIN) {
            // report but continue
            // perror("io_uring_submit");
        }

        // try to drain all ready CQEs without blocking
        while (io_uring_peek_cqe(&ring, &cqe) == 0) {
            // got a cqe
            if (cqe->res < 0) {
                // error
                // fprintf(stderr, "io_uring cqe err: %d\n", cqe->res);
            }
            completed++;
            io_uring_cqe_seen(&ring, cqe);
        }

        // If nothing ready but we still have outstanding requests, wait for at least one (block)
        if (submitted > completed) {
            if (io_uring_wait_cqe(&ring, &cqe) == 0) {
                if (cqe->res >= 0) {
                    completed++;
                } else {
                    // optional: handle error
                    completed++;
                }
                io_uring_cqe_seen(&ring, cqe);
                // after getting one, drain rest non-blocking
                while (io_uring_peek_cqe(&ring, &cqe) == 0) {
                    completed++;
                    io_uring_cqe_seen(&ring, cqe);
                }
            } else {
                // wait failed; break to avoid dead loop
            }
        } else {
            // nothing outstanding; yield CPU to be polite
            sched_yield();
        }
    }

    // cleanup
    free(offsets);
    for (int i = 0; i < t->qdepth; ++i) free(bufs[i]);
    free(bufs);
    io_uring_queue_exit(&ring);
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <mode:pread/libaio/io_uring> <threads> <qdepth> <file>\n", argv[0]);
        return 1;
    }

    char *mode = argv[1];
    int nthreads = atoi(argv[2]);
    int qdepth = atoi(argv[3]);
    char *file = argv[4];

    int fd = open(file, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    pthread_t *ths = (pthread_t*)malloc(sizeof(pthread_t) * nthreads);
    thread_arg_t *args = (thread_arg_t*)malloc(sizeof(thread_arg_t) * nthreads);

    double start = now_sec();
    for (int i = 0; i < nthreads; i++) {
        args[i].fd = fd;
        args[i].tid = i;
        args[i].qdepth = qdepth;
        args[i].ops = FILE_SIZE / BlockSize / nthreads;  // total ops divided by number of threads
        args[i].mode = mode;

        if (strcmp(mode, "psync") == 0)
            pthread_create(&ths[i], NULL, worker_pread, &args[i]);
        else if (strcmp(mode, "libaio") == 0)
            pthread_create(&ths[i], NULL, worker_libaio, &args[i]);
        else if (strcmp(mode, "io_uring") == 0)
            pthread_create(&ths[i], NULL, worker_io_uring_optimized, &args[i]);
        else {
            fprintf(stderr, "Unknown mode: %s\n", mode);
            return 1;
        }
    }

    for (int i = 0; i < nthreads; i++)
        pthread_join(ths[i], NULL);
    double end = now_sec();

    double total_ops = FILE_SIZE / BlockSize;
    double throughput = (total_ops * BlockSize) / (end - start) / (1024 * 1024);
    printf("[%s] threads=%d qdepth=%d time=%.2fs throughput=%.2f MB/s\n",
           mode, nthreads, qdepth, end - start, throughput);

    close(fd);
    free(ths);
    free(args);
    return 0;
}
