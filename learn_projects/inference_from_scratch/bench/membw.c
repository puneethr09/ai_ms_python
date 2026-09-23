// STREAM-style CPU memory bandwidth benchmark.
// Build: clang -O3 -march=native -o membw membw.c -lpthread
// Output lines: "<kernel> <threads> <GB/s>" (best of REPS runs).
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define N_ELEMS (64L * 1024 * 1024)  // 512 MiB per array, far larger than the SLC
#define REPS 15

typedef enum { K_READ, K_COPY, K_TRIAD } kernel_t;

static double *a, *b, *c;
static volatile double sink;

typedef struct {
    kernel_t kernel;
    long begin, end;
    double partial;
} job_t;

static void *run_slice(void *arg) {
    job_t *j = arg;
    switch (j->kernel) {
    case K_READ: {
        // Independent accumulators so the reduction is limited by memory, not the FP add latency chain.
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        long i = j->begin;
        for (; i + 8 <= j->end; i += 8) {
            s0 += a[i]; s1 += a[i + 1]; s2 += a[i + 2]; s3 += a[i + 3];
            s4 += a[i + 4]; s5 += a[i + 5]; s6 += a[i + 6]; s7 += a[i + 7];
        }
        for (; i < j->end; i++) s0 += a[i];
        j->partial = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
        break;
    }
    case K_COPY:
        memcpy(c + j->begin, a + j->begin, (size_t)(j->end - j->begin) * sizeof(double));
        break;
    case K_TRIAD:
        for (long i = j->begin; i < j->end; i++) a[i] = b[i] + 3.0 * c[i];
        break;
    }
    return NULL;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double best_seconds(kernel_t k, int nthreads) {
    pthread_t tids[64];
    job_t jobs[64];
    double best = 1e30;
    for (int r = 0; r < REPS; r++) {
        double t0 = now_s();
        for (int t = 0; t < nthreads; t++) {
            jobs[t] = (job_t){k, N_ELEMS * t / nthreads, N_ELEMS * (t + 1) / nthreads, 0};
            pthread_create(&tids[t], NULL, run_slice, &jobs[t]);
        }
        double total = 0;
        for (int t = 0; t < nthreads; t++) {
            pthread_join(tids[t], NULL);
            total += jobs[t].partial;
        }
        double dt = now_s() - t0;
        sink = total;
        if (dt < best) best = dt;
    }
    return best;
}

int main(int argc, char **argv) {
    int max_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (argc > 1) max_threads = atoi(argv[1]);
    if (max_threads > 64) max_threads = 64;

    a = malloc(N_ELEMS * sizeof(double));
    b = malloc(N_ELEMS * sizeof(double));
    c = malloc(N_ELEMS * sizeof(double));
    if (!a || !b || !c) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    for (long i = 0; i < N_ELEMS; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.5; }

    const double bytes = (double)N_ELEMS * sizeof(double);
    const struct { const char *name; kernel_t k; double bytes_moved; } kernels[] = {
        {"read", K_READ, bytes},
        {"copy", K_COPY, 2 * bytes},
        {"triad", K_TRIAD, 3 * bytes},
    };

    int thread_counts[] = {1, 2, 4, max_threads};
    for (size_t ki = 0; ki < 3; ki++) {
        for (size_t ti = 0; ti < 4; ti++) {
            int nt = thread_counts[ti];
            if (nt > max_threads || (ti > 0 && nt <= thread_counts[ti - 1])) continue;
            double s = best_seconds(kernels[ki].k, nt);
            printf("%s %d %.2f\n", kernels[ki].name, nt, kernels[ki].bytes_moved / s / 1e9);
            fflush(stdout);
        }
    }
    free(a); free(b); free(c);
    return 0;
}
