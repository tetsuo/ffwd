/* test_models.c - resolving a served model by its request label. */

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

int main(void) {
    test_loaded_model_lookup();
    return TEST_REPORT("models");
}
