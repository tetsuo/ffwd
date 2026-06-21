/* Parity driver for the BERT-family forward. Loads a sentence-transformers
 * checkpoint, wraps each argument with [CLS]/[SEP], runs the BERT forward
 * (mean pool + L2 normalize inside ffwd_model_encode_into), and prints
 * "<dim> <f> <f> ..." per line. check_bert_parity.py builds this with
 * `make parity-bert-driver` and compares against SentenceTransformer.encode.
 * This one needs the real model weights, so it stays a manual --model-dir
 * check rather than a hermetic stored-vector test. */
#include "internal.h"
#include "wordpiece.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL_DIR TEXT...\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];
    tok_wp_t *tok = tok_wp_load(dir);
    ffwd_model_t *model = ffwd_model_load(dir);
    if (!tok || !model) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    ffwd_workspace_t *ws = ffwd_workspace_new(model);
    const ffwd_config_t *cfg = ffwd_model_config(model);
    int dim = cfg->hidden_size;
    int cls = tok_wp_token_id(tok, "[CLS]");
    int sep = tok_wp_token_id(tok, "[SEP]");
    if (!ws || cls < 0 || sep < 0) {
        fprintf(stderr, "setup failed\n");
        return 1;
    }
    for (int a = 2; a < argc; a++) {
        int n = 0;
        int *ids = tok_wp_encode(tok, argv[a], &n);
        if (!ids && n != 0) {
            fprintf(stderr, "tokenize failed\n");
            return 1;
        }
        int *full = (int *)malloc((size_t)(n + 2) * sizeof(int));
        full[0] = cls;
        for (int i = 0; i < n; i++)
            full[1 + i] = ids[i];
        full[n + 1] = sep;
        float *emb = (float *)malloc((size_t)dim * sizeof(float));
        if (ffwd_model_encode_into(model, ws, full, n + 2, emb) != 0) {
            fprintf(stderr, "embed failed\n");
            free(ids);
            free(full);
            free(emb);
            return 1;
        }
        printf("%d", dim);
        for (int i = 0; i < dim; i++)
            printf(" %.7g", emb[i]);
        printf("\n");
        free(ids);
        free(full);
        free(emb);
    }
    ffwd_workspace_free(ws);
    ffwd_model_free(model);
    tok_wp_free(tok);
    return 0;
}
