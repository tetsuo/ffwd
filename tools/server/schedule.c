#include "server_internal.h"
#include "util.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/* ========================================================================
 * Job queue, micro-batching, and completion
 *
 * dispatch_request (server.c) enqueues one job per framed request; the
 * inference worker thread (worker_main, server.c) drains them with
 * collect_job_batch (a short first-arrival wait groups concurrent arrivals) and
 * runs process_job_group, which routes each job to its endpoint handler and
 * merges all compatible requests into one pool that execute_*_request_list
 * re-packs into token-budget batches. Finished jobs go on the done queue and
 * wake the event loop through the completion pipe; completion_cb writes each
 * response back on the loop thread, so sockets are only ever touched there.
 * ======================================================================== */

static void enqueue_done(job *j) {
    http_server *s = j->srv;
    pthread_mutex_lock(&s->mu);
    if (s->done_tail)
        s->done_tail->next = j;
    else
        s->done_head = j;
    s->done_tail = j;
    pthread_mutex_unlock(&s->mu);
    const unsigned char byte = 1;
    if (write(s->completion_pipe[1], &byte, 1) < 0) {
        /* best-effort wakeup; the event loop also drains the done queue */
    }
}

void enqueue_render_job(job *j) {
    http_server *s = j->srv;
    pthread_mutex_lock(&s->mu);
    if (s->render_tail)
        s->render_tail->next = j;
    else
        s->render_head = j;
    s->render_tail = j;
    pthread_cond_signal(&s->render_cv);
    pthread_mutex_unlock(&s->mu);
}

job *dequeue_render_job(http_server *s) {
    pthread_mutex_lock(&s->mu);
    while (!s->render_head && !s->render_stopping)
        pthread_cond_wait(&s->render_cv, &s->mu);

    job *j = s->render_head;
    if (j) {
        s->render_head = j->next;
        if (!s->render_head)
            s->render_tail = NULL;
        j->next = NULL;
    }
    pthread_mutex_unlock(&s->mu);
    return j;
}

void enqueue_raw_job(job *j) {
    http_server *s = j->srv;
    pthread_mutex_lock(&s->mu);
    if (s->raw_tail)
        s->raw_tail->next = j;
    else
        s->raw_head = j;
    s->raw_tail = j;
    pthread_cond_signal(&s->raw_cv);
    pthread_mutex_unlock(&s->mu);
}

job *dequeue_raw_job(http_server *s) {
    pthread_mutex_lock(&s->mu);
    while (!s->raw_head && !s->raw_stopping)
        pthread_cond_wait(&s->raw_cv, &s->mu);

    job *j = s->raw_head;
    if (j) {
        s->raw_head = j->next;
        if (!s->raw_head)
            s->raw_tail = NULL;
        j->next = NULL;
    }
    pthread_mutex_unlock(&s->mu);
    return j;
}

static job *pop_job_locked(http_server *s) {
    job *j = s->job_head;
    if (j) {
        s->job_head = j->next;
        if (!s->job_head)
            s->job_tail = NULL;
        j->next = NULL;
    }
    return j;
}

static int cond_wait_until_ns(pthread_cond_t *cv, pthread_mutex_t *mu, uint64_t deadline_ns) {
    uint64_t now_ns = nstime();
    if (now_ns >= deadline_ns)
        return ETIMEDOUT;

    uint64_t remaining_ns = deadline_ns - now_ns;
    struct timespec abs;
    if (clock_gettime(CLOCK_REALTIME, &abs) != 0) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        abs.tv_sec = tv.tv_sec;
        abs.tv_nsec = (long)tv.tv_usec * 1000L;
    }
    abs.tv_sec += (time_t)(remaining_ns / 1000000000u);
    abs.tv_nsec += (long)(remaining_ns % 1000000000u);
    if (abs.tv_nsec >= 1000000000L) {
        abs.tv_sec++;
        abs.tv_nsec -= 1000000000L;
    }
    return pthread_cond_timedwait(cv, mu, &abs);
}

int collect_job_batch(http_server *s, job **jobs, int max_jobs) {
    if (max_jobs <= 0)
        return 0;

    pthread_mutex_lock(&s->mu);
    while (!s->job_head && !s->stopping)
        pthread_cond_wait(&s->cv, &s->mu);

    job *first = pop_job_locked(s);
    if (!first) {
        pthread_mutex_unlock(&s->mu);
        return 0;
    }

    int n = 1;
    jobs[0] = first;
    int batchable = s->batch_size > 1 && (!strcmp(first->path, "/v1/embeddings") ||
                                          !strcmp(first->path, "/v1/contextualizedembeddings"));
    if (!batchable) {
        pthread_mutex_unlock(&s->mu);
        return n;
    }

    /* Exact item and token budgets are applied after worker-side tokenization;
     * this queue stage only defines which arrival window may be grouped. */
    uint64_t wait_ns = (uint64_t)s->batch_wait_us * 1000u;
    uint64_t deadline_ns =
        first->created_ns > UINT64_MAX - wait_ns ? UINT64_MAX : first->created_ns + wait_ns;
    while (n < max_jobs) {
        while (!s->job_head && !s->stopping && wait_ns > 0) {
            int rc = cond_wait_until_ns(&s->cv, &s->mu, deadline_ns);
            if (rc == ETIMEDOUT)
                break;
        }
        if (!s->job_head)
            break;
        if (wait_ns > 0 && s->job_head->created_ns > deadline_ns)
            break;
        jobs[n++] = pop_job_locked(s);
        if (wait_ns == 0 && !s->job_head)
            break;
    }
    pthread_mutex_unlock(&s->mu);
    return n;
}

/* True when more work is already waiting for the worker. The render placement
 * decision (handlers.c) uses this so it only hands a finished batch to the
 * renderer thread when the worker has another batch to run. */
int worker_has_pending_jobs(http_server *s) {
    if (s->worker_local_backlog)
        return 1;
    pthread_mutex_lock(&s->mu);
    int pending = s->job_head != NULL;
    pthread_mutex_unlock(&s->mu);
    return pending;
}

void enqueue_job(job *j) {
    http_server *s = j->srv;
    pthread_mutex_lock(&s->mu);
    if (s->job_tail)
        s->job_tail->next = j;
    else
        s->job_head = j;
    s->job_tail = j;
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->mu);
}

static job *pop_done(http_server *s) {
    pthread_mutex_lock(&s->mu);
    job *j = s->done_head;
    if (j) {
        s->done_head = j->next;
        if (!s->done_head)
            s->done_tail = NULL;
        j->next = NULL;
    }
    pthread_mutex_unlock(&s->mu);
    return j;
}

/* Free a tokenizer-stage request still attached to a job. In the normal flow
 * the worker moves *prep out and clears it before the job is freed; this only
 * fires if a tokenized job is discarded without being processed. */
static void job_prep_free(job *j) {
    if (!j->prep)
        return;
    if (j->prep_kind == 1)
        embedding_request_free(j->prep);
    else if (j->prep_kind == 2)
        contextual_request_free(j->prep);
    else if (j->prep_kind == 3)
        rerank_request_free(j->prep);
    free(j->prep);
    j->prep = NULL;
}

static void job_free(job *j) {
    if (!j)
        return;
    job_prep_free(j);
    job_render_free(j);
    free(j->body);
    free(j->auth);
    free(j->extra_headers);
    free(j->response);
    free(j);
}

void completion_cb(aeEventLoop *loop, int fd, void *clientData, int mask) {
    (void)loop;
    (void)mask;
    http_server *s = clientData;
    unsigned char tmp[128];
    while (read(fd, tmp, sizeof(tmp)) > 0) {
    }

    job *j;
    while ((j = pop_done(s)) != NULL) {
        client *c = j->c;
        int refs_to_drop = 1; /* job reference */
        if (!c->cancelled) {
            append_http_response_ex(c, j->status, j->content_type, j->extra_headers,
                                    j->response ? j->response : "",
                                    j->response ? j->response_len : 0);
            if (queue_write(c) != AE_OK)
                refs_to_drop += close_client_unlink(c);
        }
        client_decref_n(c, refs_to_drop);
        job_free(j);
    }
}

static void job_set_timing_header(job *j) {
    if (!j || j->extra_headers || !j->created_ns)
        return;
    uint64_t done_ns = nstime();
    uint64_t queue_ns = j->started_ns > j->created_ns ? j->started_ns - j->created_ns : 0;
    uint64_t worker_ns = j->started_ns && done_ns > j->started_ns ? done_ns - j->started_ns : 0;

    char buf[384];
    int n = snprintf(buf, sizeof(buf),
                     "Server-Timing: queue;dur=%.3f, parse;dur=%.3f, "
                     "tokenize;dur=%.3f, infer;dur=%.3f, encode;dur=%.3f, "
                     "worker;dur=%.3f\r\n",
                     ns_to_ms(queue_ns), ns_to_ms(j->parse_ns), ns_to_ms(j->tokenize_ns),
                     ns_to_ms(j->infer_ns), ns_to_ms(j->encode_ns), ns_to_ms(worker_ns));
    if (n > 0 && (size_t)n < sizeof(buf))
        j->extra_headers = xstrdup(buf);
}

void finish_job(job *j) {
    job_set_timing_header(j);
    enqueue_done(j);
}

static int merge_token_budget_accepts(int64_t used_tokens, int next_tokens, int budget) {
    if (budget <= 0 || used_tokens <= 0 || next_tokens <= 0)
        return 1;
    return used_tokens <= (int64_t)budget - next_tokens;
}

static int has_unfinished_jobs(const int *done, int n_jobs) {
    for (int i = 0; i < n_jobs; i++) {
        if (!done[i])
            return 1;
    }
    return 0;
}

static int parse_job_root(job *j, cJSON **out_root) {
    cJSON *detail = cJSON_CreateArray();
    cJSON *root = parse_json_body(j, detail);
    if (!root) {
        job_set_422(j, detail);
        cJSON_Delete(detail);
        *out_root = NULL;
        return -1;
    }
    cJSON_Delete(detail);
    *out_root = root;
    return 0;
}

static void process_unknown_job(job *j) {
    cJSON *root = NULL;
    if (!j->started_ns)
        j->started_ns = nstime();
    uint64_t t0 = nstime();
    int rc = parse_job_root(j, &root);
    j->parse_ns += nstime() - t0;
    if (rc != 0)
        return;
    job_set_error(j, 404, "unknown endpoint", "invalid_request_error");
    cJSON_Delete(root);
}

/* Parse and tokenize one job into a heap request attached to j->prep, which the
 * worker then moves into its request array. Runs on the tokenizer thread, or on
 * the worker itself for a request that reached the job queue untokenized. An
 * unknown endpoint leaves prep_kind 0 (process_unknown_job handles it); a parse
 * failure leaves prep NULL with the 422 already set, so the worker finishes it
 * as not-ready. The tokenized flag guards against tokenizing twice. */
void tokenize_job(http_server *s, job *j) {
    j->tokenized = 1;
    if (!j->started_ns)
        j->started_ns = nstime();
    if (!strcmp(j->path, "/v1/embeddings"))
        j->prep_kind = 1;
    else if (!strcmp(j->path, "/v1/contextualizedembeddings"))
        j->prep_kind = 2;
    else if (!strcmp(j->path, "/v1/rerank"))
        j->prep_kind = 3;
    else
        return;

    cJSON *root = NULL;
    uint64_t t0 = nstime();
    int parse_rc = parse_job_root(j, &root);
    j->parse_ns += nstime() - t0;
    if (parse_rc != 0)
        return;

    if (j->prep_kind == 1) {
        embedding_request *r = xcalloc(1, sizeof(*r));
        prepare_embedding_request(j, root, s, r);
        j->prep = r;
    } else if (j->prep_kind == 2) {
        contextual_request *r = xcalloc(1, sizeof(*r));
        prepare_contextual_request(j, root, s, r);
        j->prep = r;
    } else {
        rerank_request *r = xcalloc(1, sizeof(*r));
        prepare_rerank_request(j, root, s, r);
        j->prep = r;
    }
}

void process_job_group(http_server *s, job **jobs, int n_jobs) {
    embedding_request *std_reqs = xcalloc((size_t)n_jobs, sizeof(*std_reqs));
    contextual_request *ctx_reqs = xcalloc((size_t)n_jobs, sizeof(*ctx_reqs));
    rerank_request *rerank_reqs = xcalloc((size_t)n_jobs, sizeof(*rerank_reqs));
    int *kind = xcalloc((size_t)n_jobs, sizeof(*kind));
    int *done = xcalloc((size_t)n_jobs, sizeof(*done));
    embedding_request **std_group = xmalloc((size_t)n_jobs * sizeof(*std_group));
    contextual_request **ctx_group = xmalloc((size_t)n_jobs * sizeof(*ctx_group));

    /* Tokenize any job that reached the queue untokenized (dispatched inline
     * past the tokenizer thread), then move each prepared request out of its job
     * into the contiguous array the grouping below indexes. The move transfers
     * the heap pointers; freeing the wrapper leaves the array slot owning them,
     * freed once at the end. */
    for (int i = 0; i < n_jobs; i++) {
        if (!jobs[i]->tokenized)
            tokenize_job(s, jobs[i]);
        kind[i] = jobs[i]->prep_kind;
        if (jobs[i]->prep) {
            if (kind[i] == 1)
                std_reqs[i] = *(embedding_request *)jobs[i]->prep;
            else if (kind[i] == 2)
                ctx_reqs[i] = *(contextual_request *)jobs[i]->prep;
            else if (kind[i] == 3)
                rerank_reqs[i] = *(rerank_request *)jobs[i]->prep;
            free(jobs[i]->prep);
            jobs[i]->prep = NULL;
        }
    }

    for (int i = 0; i < n_jobs; i++) {
        if (done[i])
            continue;

        if (!kind[i]) {
            process_unknown_job(jobs[i]);
            finish_job(jobs[i]);
            done[i] = 1;
            continue;
        }

        if ((kind[i] == 1 && !std_reqs[i].ready) || (kind[i] == 2 && !ctx_reqs[i].ready) ||
            (kind[i] == 3 && !rerank_reqs[i].ready)) {
            finish_job(jobs[i]);
            done[i] = 1;
            continue;
        }

        if (kind[i] == 3) {
            execute_rerank_request(&rerank_reqs[i]);
            finish_job(jobs[i]);
            done[i] = 1;
        } else if (kind[i] == 1) {
            /* Pool every compatible ready request in this window into one list.
             * execute_embedding_request_list length-sorts the inputs and packs
             * them into token-budget batches, so mixed client batch shapes fill
             * the GPU batch instead of each request running on a partly-empty
             * launch of its own. The merged execution unit is also capped by
             * the configured token budget so one arrival window cannot become
             * an unbounded host/output buffer before any response can finish. */
            int group_n = 1;
            int64_t group_tokens = std_reqs[i].total_tokens;
            std_group[0] = &std_reqs[i];
            for (int k = i + 1; k < n_jobs; k++) {
                if (!done[k] && kind[k] == 1 && std_reqs[k].ready &&
                    embedding_request_compatible(&std_reqs[i], &std_reqs[k]) &&
                    merge_token_budget_accepts(group_tokens, std_reqs[k].total_tokens,
                                               s->max_batch_tokens)) {
                    std_group[group_n++] = &std_reqs[k];
                    group_tokens += std_reqs[k].total_tokens;
                }
            }
            for (int k = 0; k < group_n; k++) {
                int idx = (int)(std_group[k] - std_reqs);
                done[idx] = 1;
            }
            s->worker_local_backlog = has_unfinished_jobs(done, n_jobs);
            execute_embedding_request_list(std_group, group_n);
            for (int k = 0; k < group_n; k++) {
                int idx = (int)(std_group[k] - std_reqs);
                if (jobs[idx]->render_kind)
                    enqueue_render_job(jobs[idx]);
                else
                    finish_job(jobs[idx]);
            }
            s->worker_local_backlog = 0;
        } else {
            int group_n = 1;
            int64_t group_tokens = ctx_reqs[i].total_tokens;
            ctx_group[0] = &ctx_reqs[i];
            for (int k = i + 1; k < n_jobs; k++) {
                if (!done[k] && kind[k] == 2 && ctx_reqs[k].ready &&
                    contextual_request_compatible(&ctx_reqs[i], &ctx_reqs[k]) &&
                    merge_token_budget_accepts(group_tokens, ctx_reqs[k].total_tokens,
                                               s->max_batch_tokens)) {
                    ctx_group[group_n++] = &ctx_reqs[k];
                    group_tokens += ctx_reqs[k].total_tokens;
                }
            }
            for (int k = 0; k < group_n; k++) {
                int idx = (int)(ctx_group[k] - ctx_reqs);
                done[idx] = 1;
            }
            execute_contextual_request_list(ctx_group, group_n);
            for (int k = 0; k < group_n; k++) {
                int idx = (int)(ctx_group[k] - ctx_reqs);
                finish_job(jobs[idx]);
            }
        }
    }

    for (int i = 0; i < n_jobs; i++) {
        embedding_request_free(&std_reqs[i]);
        contextual_request_free(&ctx_reqs[i]);
        rerank_request_free(&rerank_reqs[i]);
    }
    free(ctx_group);
    free(std_group);
    free(done);
    free(kind);
    free(rerank_reqs);
    free(ctx_reqs);
    free(std_reqs);
}
