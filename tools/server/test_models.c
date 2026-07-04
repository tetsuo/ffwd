/* test_models.c - resolving a served model by its request label, and the
 * per-input token cap with its truncation helper. */

#include "server_internal.h"
#include "test_util.h"

#include <string.h>

static void test_loaded_model_lookup(void) {
    model_info info;
    memset(&info, 0, sizeof(info));
    info.id = "served-label";
    loaded_model models[1];
    memset(models, 0, sizeof(models));
    models[0].info = &info;
    http_server s;
    memset(&s, 0, sizeof(s));
    s.models = models;
    s.n_models = 1;
    TEST_ASSERT(loaded_model_for_label(&s, "served-label") == &models[0]);
    TEST_ASSERT(loaded_model_for_label(&s, "compiled-name") == NULL);
}

static void test_model_item_token_cap(void) {
    model_info info;
    memset(&info, 0, sizeof(info));

    /* No model cap: the API item cap applies, then the batch budget. */
    TEST_ASSERT(model_item_token_cap(&info, 0) == FFWD_API_MAX_ITEM_TOKENS);
    TEST_ASSERT(model_item_token_cap(&info, 16384) == 16384);

    /* Model cap below both: it wins. */
    info.max_seq_tokens = 512;
    TEST_ASSERT(model_item_token_cap(&info, 16384) == 512);

    /* Batch budget below the model cap: the budget wins. */
    info.max_seq_tokens = 32768;
    TEST_ASSERT(model_item_token_cap(&info, 16384) == 16384);
    TEST_ASSERT(model_item_token_cap(&info, 0) == 32768);
}

static void test_truncate_token_buf(void) {
    int ids[6] = {101, 7, 8, 9, 10, 102};
    token_buf t = {.ids = ids, .n_tokens = 6};

    /* Under the cap: untouched. */
    truncate_token_buf(&t, 6, 1);
    TEST_ASSERT(t.n_tokens == 6 && ids[5] == 102);

    /* keep_tail moves the final special token to the new end. */
    truncate_token_buf(&t, 4, 1);
    TEST_ASSERT(t.n_tokens == 4);
    TEST_ASSERT(ids[0] == 101 && ids[1] == 7 && ids[2] == 8 && ids[3] == 102);

    /* Plain cut when the layout has no required tail. */
    truncate_token_buf(&t, 2, 0);
    TEST_ASSERT(t.n_tokens == 2);
    TEST_ASSERT(ids[0] == 101 && ids[1] == 7);
}

int main(void) {
    test_loaded_model_lookup();
    test_model_item_token_cap();
    test_truncate_token_buf();
    return TEST_REPORT("models");
}
