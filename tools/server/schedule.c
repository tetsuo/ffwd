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
 * dispatch_request (http.c) enqueues one job per framed request; the
 * inference worker thread (worker_main, server.c) drains them with
 * collect_job_batch (a short first-arrival wait groups concurrent arrivals) and
 * runs process_job_group, which routes each job to its endpoint handler and
 * batches compatible ones. Finished jobs go on the done queue and wake the event
 * loop through the completion pipe; completion_cb writes each response back on
 * the loop thread, so sockets are only ever touched there.
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

static void job_free(job *j) {
    if (!j)
        return;
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

void process_job_group(http_server *s, job **jobs, int n_jobs) {
    embedding_request *std_reqs = xcalloc((size_t)n_jobs, sizeof(*std_reqs));
    contextual_request *ctx_reqs = xcalloc((size_t)n_jobs, sizeof(*ctx_reqs));
    rerank_request *rerank_reqs = xcalloc((size_t)n_jobs, sizeof(*rerank_reqs));
    int *kind = xcalloc((size_t)n_jobs, sizeof(*kind));
    int *done = xcalloc((size_t)n_jobs, sizeof(*done));
    embedding_request **std_group = xmalloc((size_t)n_jobs * sizeof(*std_group));
    contextual_request **ctx_group = xmalloc((size_t)n_jobs * sizeof(*ctx_group));

    for (int i = 0; i < n_jobs; i++) {
        cJSON *root = NULL;
        if (!jobs[i]->started_ns)
            jobs[i]->started_ns = nstime();
        if (!strcmp(jobs[i]->path, "/v1/embeddings")) {
            kind[i] = 1;
        } else if (!strcmp(jobs[i]->path, "/v1/contextualizedembeddings")) {
            kind[i] = 2;
        } else if (!strcmp(jobs[i]->path, "/v1/rerank")) {
            kind[i] = 3;
        } else {
            continue;
        }
        uint64_t t0 = nstime();
        int parse_rc = parse_job_root(jobs[i], &root);
        jobs[i]->parse_ns += nstime() - t0;
        if (parse_rc != 0)
            continue;
        if (kind[i] == 1)
            prepare_embedding_request(jobs[i], root, s, &std_reqs[i]);
        else if (kind[i] == 2)
            prepare_contextual_request(jobs[i], root, s, &ctx_reqs[i]);
        else
            prepare_rerank_request(jobs[i], root, s, &rerank_reqs[i]);
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
            int group_n = 1;
            int group_inputs = std_reqs[i].n_inputs;
            int group_tokens = std_reqs[i].total_tokens;
            std_group[0] = &std_reqs[i];
            for (int k = i + 1; k < n_jobs; k++) {
                if (!done[k] && kind[k] == 1 && std_reqs[k].ready &&
                    embedding_request_compatible(&std_reqs[i], &std_reqs[k]) &&
                    embedding_request_fits_group(&std_reqs[k], group_inputs, group_tokens,
                                                 s->batch_size, s->max_batch_tokens)) {
                    std_group[group_n++] = &std_reqs[k];
                    group_inputs += std_reqs[k].n_inputs;
                    group_tokens += std_reqs[k].total_tokens;
                }
            }
            execute_embedding_request_list(std_group, group_n);
            for (int k = 0; k < group_n; k++) {
                int idx = (int)(std_group[k] - std_reqs);
                if (jobs[idx]->render_kind)
                    enqueue_render_job(jobs[idx]);
                else
                    finish_job(jobs[idx]);
                done[idx] = 1;
            }
        } else {
            int group_n = 1;
            int group_docs = ctx_reqs[i].n_docs;
            int group_tokens = ctx_reqs[i].total_tokens;
            ctx_group[0] = &ctx_reqs[i];
            for (int k = i + 1; k < n_jobs; k++) {
                if (!done[k] && kind[k] == 2 && ctx_reqs[k].ready &&
                    contextual_request_compatible(&ctx_reqs[i], &ctx_reqs[k]) &&
                    contextual_request_fits_group(&ctx_reqs[k], group_docs, group_tokens,
                                                  s->batch_size, s->max_batch_tokens)) {
                    ctx_group[group_n++] = &ctx_reqs[k];
                    group_docs += ctx_reqs[k].n_docs;
                    group_tokens += ctx_reqs[k].total_tokens;
                }
            }
            execute_contextual_request_list(ctx_group, group_n);
            for (int k = 0; k < group_n; k++) {
                int idx = (int)(ctx_group[k] - ctx_reqs);
                finish_job(jobs[idx]);
                done[idx] = 1;
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
