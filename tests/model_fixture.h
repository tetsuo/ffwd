/* tests/model_fixture.h - shared model+tokenizer fixture writer. */

#ifndef MODEL_FIXTURE_H
#define MODEL_FIXTURE_H

#include "tiny_model.h"
#include "tok_fixture.h"

#include <string.h>

typedef enum {
    MF_MODEL_BASE,
    MF_MODEL_QWEN3,
    MF_MODEL_QWEN2,
} mf_model_kind_t;

static inline tm_dims_t mf_default_dims(void) {
    tm_dims_t d = {4, 2, 1, 2, 8, TF_VOCAB_SIZE};
    return d;
}

static inline int mf_valid_dtype(const char *dtype) {
    return !strcmp(dtype, "F32") || !strcmp(dtype, "BF16") || !strcmp(dtype, "F16");
}

static inline int mf_parse_model_kind(const char *name, mf_model_kind_t *kind) {
    if (!strcmp(name, "base")) {
        *kind = MF_MODEL_BASE;
        return 0;
    }
    if (!strcmp(name, "qwen3")) {
        *kind = MF_MODEL_QWEN3;
        return 0;
    }
    if (!strcmp(name, "qwen2")) {
        *kind = MF_MODEL_QWEN2;
        return 0;
    }
    return -1;
}

static inline int mf_write_model(const char *dir,
                                 const char *dtype,
                                 const tm_dims_t *dims,
                                 mf_model_kind_t kind,
                                 int eos_token_id,
                                 int qwen2_zero_bias) {
    switch (kind) {
    case MF_MODEL_BASE:
        return tm_write_model_dims(dir, dtype, dims);
    case MF_MODEL_QWEN3:
        return tm_write_qwen3_model_dims(dir, dtype, dims, eos_token_id);
    case MF_MODEL_QWEN2:
        return tm_write_qwen2_model_dims(dir, dtype, dims, eos_token_id, qwen2_zero_bias);
    }
    return -1;
}

static inline int mf_write_fixture(const char *dir,
                                   const char *dtype,
                                   const tm_dims_t *dims,
                                   mf_model_kind_t kind,
                                   int eos_token_id,
                                   int write_vocab,
                                   int qwen2_zero_bias) {
    if (!mf_valid_dtype(dtype))
        return -1;
    if (write_vocab && tf_write_vocab(dir) != 0)
        return -1;
    return mf_write_model(dir, dtype, dims, kind, eos_token_id, qwen2_zero_bias);
}

#endif /* MODEL_FIXTURE_H */
