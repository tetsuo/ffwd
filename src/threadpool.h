#ifndef EMBED_THREADPOOL_H
#define EMBED_THREADPOOL_H

/* Persistent worker pool for the CPU math kernels: a fixed set of threads that
 * sleep until parallel_for() hands them a callback, plus the CPU-count helper
 * that sizes the pool (cgroup-quota aware). Separated from the math so the
 * kernels depend only on the parallel_for / embed_num_threads interface, not on
 * the pool's internals. kernels.h includes this, so kernel callers need no
 * extra include. */

#ifndef EMBED_API
#if defined(__GNUC__)
#define EMBED_API __attribute__((visibility("default")))
#else
#define EMBED_API
#endif
#endif

/* Worker callback: (thread index, total threads, user arg). */
typedef void (*parallel_fn_t)(int tid, int n_threads, void *arg);

/* Set the persistent pool size (default 1). Creates/rebuilds the pool; call
 * before inference. */
EMBED_API void embed_set_threads(int n);

/* Available CPU cores, capped to a cgroup CPU quota when one is tighter. */
EMBED_API int embed_get_num_cpus(void);

/* Current pool size (1 == serial). Kernels gate parallelism on this. */
int embed_num_threads(void);

/* Run fn across the pool; the calling thread participates as tid 0. Blocks
 * until every worker has finished. */
void parallel_for(parallel_fn_t fn, void *arg);

#endif /* EMBED_THREADPOOL_H */
