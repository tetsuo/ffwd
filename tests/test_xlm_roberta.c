/* tests/test_xlm_roberta.c - hermetic XLM-R/RoBERTa bridge checks. */

#include "embed.h"
#include "tiny_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int write_fixture(const char *dir) {
    const int hidden = 4, heads = 2, inter = 8, vocab = 16, max_pos = 8;
    char path[2048];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE *f = fopen(path, "w");
    if (!f)
        return -1;
    fprintf(f,
            "{\"model_type\":\"xlm-roberta\",\"hidden_size\":%d,"
            "\"num_hidden_layers\":1,\"num_attention_heads\":%d,"
            "\"intermediate_size\":%d,\"vocab_size\":%d,"
            "\"max_position_embeddings\":%d,\"layer_norm_eps\":1e-5,"
            "\"pad_token_id\":1,\"type_vocab_size\":1}",
            hidden, heads, inter, vocab, max_pos);
    fclose(f);

    const tm_spec_t specs[] = {
        {"embeddings.word_embeddings.weight", vocab, hidden},
        {"embeddings.position_embeddings.weight", max_pos, hidden},
        {"embeddings.token_type_embeddings.weight", 1, hidden},
        {"embeddings.LayerNorm.weight", hidden, 0},
        {"embeddings.LayerNorm.bias", hidden, 0},
        {"encoder.layer.0.attention.self.query.weight", hidden, hidden},
        {"encoder.layer.0.attention.self.query.bias", hidden, 0},
        {"encoder.layer.0.attention.self.key.weight", hidden, hidden},
        {"encoder.layer.0.attention.self.key.bias", hidden, 0},
        {"encoder.layer.0.attention.self.value.weight", hidden, hidden},
        {"encoder.layer.0.attention.self.value.bias", hidden, 0},
        {"encoder.layer.0.attention.output.dense.weight", hidden, hidden},
        {"encoder.layer.0.attention.output.dense.bias", hidden, 0},
        {"encoder.layer.0.attention.output.LayerNorm.weight", hidden, 0},
        {"encoder.layer.0.attention.output.LayerNorm.bias", hidden, 0},
        {"encoder.layer.0.intermediate.dense.weight", inter, hidden},
        {"encoder.layer.0.intermediate.dense.bias", inter, 0},
        {"encoder.layer.0.output.dense.weight", hidden, inter},
        {"encoder.layer.0.output.dense.bias", hidden, 0},
        {"encoder.layer.0.output.LayerNorm.weight", hidden, 0},
        {"encoder.layer.0.output.LayerNorm.bias", hidden, 0},
    };
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    return tm_write_safetensors_with_prefix(path, "F32", specs,
                                            (int)(sizeof(specs) / sizeof(specs[0])),
                                            "roberta.");
}

int main(void) {
    const char *tmp = getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp";
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/embed-xlm-roberta-test-XXXXXX", tmp);
    if (!mkdtemp(dir) || write_fixture(dir) != 0) {
        fprintf(stderr, "fixture creation failed\n");
        return 2;
    }

    embed_model_t *model = embed_model_load(dir);
    embed_workspace_t *ws = embed_workspace_new(model);
    const embed_config_t *cfg = embed_model_config(model);
    if (!model || !ws || !cfg) {
        fprintf(stderr, "model setup failed\n");
        return 1;
    }
    if (cfg->family != EMBED_FAMILY_BERT || cfg->position_id_offset != 2 ||
        cfg->type_vocab_size != 1 || cfg->pooling_mode != EMBED_POOL_MEAN ||
        cfg->attention_mode != EMBED_ATTENTION_BIDIRECTIONAL) {
        fprintf(stderr, "XLM-R config bridge wrong\n");
        return 1;
    }

    float out[4];
    int ids_ok[] = {0, 4, 5, 6, 7, 2};
    if (embed_model_encode_into(model, ws, ids_ok, 6, out) != 0) {
        fprintf(stderr, "XLM-R forward failed at max valid position\n");
        return 1;
    }
    int ids_too_long[] = {0, 4, 5, 6, 7, 8, 2};
    if (embed_model_encode_into(model, ws, ids_too_long, 7, out) == 0) {
        fprintf(stderr, "XLM-R position overflow was accepted\n");
        return 1;
    }

    embed_workspace_free(ws);
    embed_model_free(model);
    puts("ok: XLM-R config, roberta prefix, type_vocab_size=1, position offset");
    return 0;
}
