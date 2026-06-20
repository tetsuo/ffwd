/*
 * tp.c - CPU thread pool
 */

#if defined(__linux__) && !defined(_GNU_SOURCE)
#    define _GNU_SOURCE
#endif

#include "threadpool.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#    include <sys/sysctl.h>
#else
#    ifdef __linux__
#        include <sched.h>
#    endif
#    include <unistd.h>
#endif

#if defined(USE_OPENBLAS)
void openblas_set_num_threads(int num_threads);
#endif

#define TP_MAX_THREADS 16

static struct {
    pthread_t threads[TP_MAX_THREADS - 1];
    int tids[TP_MAX_THREADS - 1];
    int n_threads;
    int shutdown;

    parallel_fn_t fn;
    void *arg;
    uint64_t generation;

    pthread_mutex_t mutex;
    pthread_cond_t cond_work;
    pthread_cond_t cond_done;
    int n_done;
} tp = {
    .n_threads = 1,
    .shutdown = 0,
    .generation = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond_work = PTHREAD_COND_INITIALIZER,
    .cond_done = PTHREAD_COND_INITIALIZER,
};

static void *worker_loop(void *arg) {
    int tid = *(int *)arg;
    uint64_t my_gen = 0;

    for (;;) {
        pthread_mutex_lock(&tp.mutex);
        while (tp.generation == my_gen && !tp.shutdown)
            pthread_cond_wait(&tp.cond_work, &tp.mutex);
        if (tp.shutdown) {
            pthread_mutex_unlock(&tp.mutex);
            return NULL;
        }
        my_gen = tp.generation;
        parallel_fn_t fn = tp.fn;
        void *a = tp.arg;
        int nt = tp.n_threads;
        pthread_mutex_unlock(&tp.mutex);

        fn(tid, nt, a);

        pthread_mutex_lock(&tp.mutex);
        if (++tp.n_done >= tp.n_threads - 1)
            pthread_cond_signal(&tp.cond_done);
        pthread_mutex_unlock(&tp.mutex);
    }
}

void tp_set_threads(int n) {
    if (n < 1)
        n = 1;
    if (n > TP_MAX_THREADS)
        n = TP_MAX_THREADS;

#if defined(USE_OPENBLAS)
    /* OpenBLAS owns the heavy F32 GEMM parallelism on Linux. Keep the
     * auxiliary pool serial to avoid nested thread contention. */
    int blas_threads = n;
    openblas_set_num_threads(blas_threads);
    n = 1;
#endif

    /* Shutdown existing workers */
    if (tp.n_threads > 1) {
        pthread_mutex_lock(&tp.mutex);
        tp.shutdown = 1;
        pthread_cond_broadcast(&tp.cond_work);
        pthread_mutex_unlock(&tp.mutex);
        for (int i = 0; i < tp.n_threads - 1; i++)
            pthread_join(tp.threads[i], NULL);
        tp.shutdown = 0;
        tp.generation = 0;
    }

    tp.n_threads = n;
    if (n <= 1)
        return;

    for (int i = 0; i < n - 1; i++) {
        tp.tids[i] = i + 1;
        pthread_create(&tp.threads[i], NULL, worker_loop, &tp.tids[i]);
    }
}

#ifndef __APPLE__
/* CPU bandwidth this process may actually use, in whole cores, or 0 if there
 * is no cgroup limit. Containers (Docker --cpus, Kubernetes, RunPod) expose
 * every host CPU through sysconf()/cpuset, but the CFS quota caps the CPU time
 * the process really gets. Sizing the BLAS/thread pool to the host count then
 * oversubscribes that quota: the threads spend the period's budget early and
 * the scheduler throttles them for the rest of each period, which adds large,
 * bursty stalls (e.g. a 16-thread default against a 5.1-core quota ran the
 * 0.6B model ~2x slower than a 5-thread pool). Floor, not ceil, so a fully
 * busy pool stays under the quota and never trips throttling. */
static int cgroup_cpu_quota(void) {
    /* cgroup v2: "/sys/fs/cgroup/cpu.max" is "<quota> <period>", or
     * "max <period>" when unlimited. */
    FILE *f = fopen("/sys/fs/cgroup/cpu.max", "r");
    if (f) {
        char quota[32];
        long period = 0;
        int got = fscanf(f, "%31s %ld", quota, &period);
        fclose(f);
        if (got == 2 && period > 0 && strcmp(quota, "max") != 0) {
            long q = atol(quota);
            if (q > 0)
                return q / period < 1 ? 1 : (int)(q / period);
        }
    }
    /* cgroup v1: separate files; quota -1 means unlimited. */
    long q = -1, period = 0;
    f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r");
    if (f) {
        if (fscanf(f, "%ld", &q) != 1)
            q = -1;
        fclose(f);
    }
    f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r");
    if (f) {
        if (fscanf(f, "%ld", &period) != 1)
            period = 0;
        fclose(f);
    }
    if (q > 0 && period > 0)
        return q / period < 1 ? 1 : (int)(q / period);
    return 0;
}
#endif

int tp_get_num_cpus(void) {
#ifdef __APPLE__
    int n = 0;
    size_t len = sizeof(n);
    sysctlbyname("hw.ncpu", &n, &len, NULL, 0);
    return n > 0 ? n : 1;
#else
    int online = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (online < 1)
        online = 1;
#    ifdef __linux__
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int affinity = CPU_COUNT(&set);
        if (affinity >= 1 && affinity < online)
            online = affinity;
    }
#    endif
    /* Honor a cgroup CPU quota when it is tighter than the visible core count,
     * so the default thread pool does not oversubscribe a CPU-limited
     * container and get throttled. */
    int quota = cgroup_cpu_quota();
    if (quota >= 1 && quota < online)
        return quota;
    return online;
#endif
}

int tp_num_threads(void) { return tp.n_threads; }

/* Dispatch work to all threads; main thread is tid=0 */
void parallel_for(parallel_fn_t fn, void *arg) {
    if (tp.n_threads <= 1) {
        fn(0, 1, arg);
        return;
    }

    pthread_mutex_lock(&tp.mutex);
    tp.fn = fn;
    tp.arg = arg;
    tp.n_done = 0;
    tp.generation++;
    pthread_cond_broadcast(&tp.cond_work);
    pthread_mutex_unlock(&tp.mutex);

    fn(0, tp.n_threads, arg);

    pthread_mutex_lock(&tp.mutex);
    while (tp.n_done < tp.n_threads - 1)
        pthread_cond_wait(&tp.cond_done, &tp.mutex);
    pthread_mutex_unlock(&tp.mutex);
}
