#ifndef TP_THREADPOOL_H
#define TP_THREADPOOL_H

/* Persistent worker pool for CPU kernels.
 * Uses a fixed thread set that sleeps until parallel_for() gives it a callback.
 * Also provides the cgroup-quota-aware CPU-count helper used to size the pool.
 *
 * Kept separate from the kernels so they depend only on parallel_for and
 * tp_num_threads, not pool internals.
 *
 * kernels.h includes this, so kernel callers need no extra include. */

/* Worker callback: (thread index, total threads, user arg). */
typedef void (*parallel_fn_t)(int tid, int n_threads, void *arg);

/* Set the persistent pool size (default 1). Creates/rebuilds the pool; call
 * before inference. */
void tp_set_threads(int n);

/* Available CPU cores, capped to a cgroup CPU quota when one is tighter. */
int tp_get_num_cpus(void);

/* Current pool size (1 == serial). Kernels gate parallelism on this. */
int tp_num_threads(void);

/* Run fn across the pool; the calling thread participates as tid 0. Blocks
 * until every worker has finished. */
void parallel_for(parallel_fn_t fn, void *arg);

#endif /* TP_THREADPOOL_H */
