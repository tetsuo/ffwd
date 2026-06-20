/* test_schedule.c - job queue micro-batching: the batch-wait deadline and the
 * zero-wait fast path through enqueue_job / collect_job_batch. */

#include "server_internal.h"
#include "test_util.h"
#include "util.h" /* nstime */

#include <pthread.h>
#include <stdio.h>
#include <string.h>

static void test_collect_job_batch_deadline(void) {
    http_server s;
    memset(&s, 0, sizeof(s));
    s.batch_size = 32;
    s.batch_wait_us = 1000;
    pthread_mutex_init(&s.mu, NULL);
    pthread_cond_init(&s.cv, NULL);

    uint64_t first_ns = nstime() - 5000000u;
    job first = {.srv = &s, .created_ns = first_ns};
    job before_deadline = {.srv = &s, .created_ns = first_ns + 500000u};
    job after_deadline = {.srv = &s, .created_ns = first_ns + 2000000u};
    snprintf(first.path, sizeof(first.path), "%s", "/v1/embeddings");
    snprintf(before_deadline.path, sizeof(before_deadline.path), "%s", "/v1/embeddings");
    snprintf(after_deadline.path, sizeof(after_deadline.path), "%s", "/v1/embeddings");
    enqueue_job(&first);
    enqueue_job(&before_deadline);
    enqueue_job(&after_deadline);

    job *batch[3] = {0};
    TEST_ASSERT(collect_job_batch(&s, batch, 3) == 2);
    TEST_ASSERT(batch[0] == &first);
    TEST_ASSERT(batch[1] == &before_deadline);
    TEST_ASSERT(s.job_head == &after_deadline);
    TEST_ASSERT(s.job_tail == &after_deadline);

    s.job_head = NULL;
    s.job_tail = NULL;
    pthread_cond_destroy(&s.cv);
    pthread_mutex_destroy(&s.mu);
}

static void test_collect_job_batch_zero_wait(void) {
    http_server s;
    memset(&s, 0, sizeof(s));
    s.batch_size = 32;
    pthread_mutex_init(&s.mu, NULL);
    pthread_cond_init(&s.cv, NULL);

    job first = {.srv = &s, .created_ns = nstime()};
    job second = {.srv = &s, .created_ns = nstime()};
    snprintf(first.path, sizeof(first.path), "%s", "/v1/embeddings");
    snprintf(second.path, sizeof(second.path), "%s", "/v1/embeddings");
    enqueue_job(&first);
    enqueue_job(&second);

    job *batch[2] = {0};
    TEST_ASSERT(collect_job_batch(&s, batch, 2) == 2);
    TEST_ASSERT(batch[0] == &first);
    TEST_ASSERT(batch[1] == &second);
    TEST_ASSERT(s.job_head == NULL);
    TEST_ASSERT(s.job_tail == NULL);

    pthread_cond_destroy(&s.cv);
    pthread_mutex_destroy(&s.mu);
}

int main(void) {
    test_collect_job_batch_deadline();
    test_collect_job_batch_zero_wait();
    return TEST_REPORT("schedule");
}
