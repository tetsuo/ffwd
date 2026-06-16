#include "safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>

/* ========================================================================
 * Minimal JSON parser for safetensors header
 * ======================================================================== */

static void skip_whitespace(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t')
        (*p)++;
}

static int parse_string(const char **p, char *out, size_t max_len) {
    skip_whitespace(p);
    if (**p != '"')
        return -1;
    (*p)++;
    size_t i = 0;
    while (**p && **p != '"' && i < max_len - 1) {
        if (**p == '\\') {
            (*p)++;
            if (**p == 'n')
                out[i++] = '\n';
            else if (**p == 't')
                out[i++] = '\t';
            else if (**p == '"')
                out[i++] = '"';
            else if (**p == '\\')
                out[i++] = '\\';
            else
                out[i++] = **p;
        } else {
            out[i++] = **p;
        }
        (*p)++;
    }
    out[i] = '\0';
    if (**p != '"')
        return -1;
    (*p)++;
    return 0;
}

static int parse_int(const char **p, int64_t *out) {
    skip_whitespace(p);
    int64_t val = 0;
    int neg = 0;
    if (**p == '-') {
        neg = 1;
        (*p)++;
    }
    if (**p < '0' || **p > '9')
        return -1;
    while (**p >= '0' && **p <= '9') {
        val = val * 10 + (**p - '0');
        (*p)++;
    }
    *out = neg ? -val : val;
    return 0;
}

static safetensor_dtype_t parse_dtype(const char *s) {
    if (strcmp(s, "F32") == 0)
        return DTYPE_F32;
    if (strcmp(s, "F16") == 0)
        return DTYPE_F16;
    if (strcmp(s, "BF16") == 0)
        return DTYPE_BF16;
    if (strcmp(s, "I32") == 0)
        return DTYPE_I32;
    if (strcmp(s, "I64") == 0)
        return DTYPE_I64;
    if (strcmp(s, "BOOL") == 0)
        return DTYPE_BOOL;
    return DTYPE_UNKNOWN;
}

static int parse_tensor_entry(const char **p, safetensor_t *t) {
    skip_whitespace(p);
    if (**p != '{')
        return -1;
    (*p)++;

    t->dtype = DTYPE_UNKNOWN;
    t->ndim = 0;
    t->data_offset = 0;
    t->data_size = 0;

    while (**p && **p != '}') {
        skip_whitespace(p);
        if (**p == ',') {
            (*p)++;
            continue;
        }

        char key[64];
        if (parse_string(p, key, sizeof(key)) != 0)
            return -1;
        skip_whitespace(p);
        if (**p != ':')
            return -1;
        (*p)++;
        skip_whitespace(p);

        if (strcmp(key, "dtype") == 0) {
            char dtype_str[32];
            if (parse_string(p, dtype_str, sizeof(dtype_str)) != 0)
                return -1;
            t->dtype = parse_dtype(dtype_str);
        } else if (strcmp(key, "shape") == 0) {
            if (**p != '[')
                return -1;
            (*p)++;
            t->ndim = 0;
            while (**p && **p != ']') {
                skip_whitespace(p);
                if (**p == ',') {
                    (*p)++;
                    continue;
                }
                if (t->ndim >= 8)
                    return -1;
                if (parse_int(p, &t->shape[t->ndim]) != 0)
                    return -1;
                t->ndim++;
            }
            if (**p != ']')
                return -1;
            (*p)++;
        } else if (strcmp(key, "data_offsets") == 0) {
            if (**p != '[')
                return -1;
            (*p)++;
            skip_whitespace(p);
            int64_t start_i;
            if (parse_int(p, &start_i) != 0)
                return -1;
            skip_whitespace(p);
            if (**p != ',')
                return -1;
            (*p)++;
            skip_whitespace(p);
            int64_t end_i;
            if (parse_int(p, &end_i) != 0)
                return -1;
            if (start_i < 0 || end_i < start_i)
                return -1;
            t->data_offset = (size_t)start_i;
            t->data_size = (size_t)(end_i - start_i);
            skip_whitespace(p);
            if (**p != ']')
                return -1;
            (*p)++;
        } else {
            /* Skip unknown value */
            if (**p == '"') {
                (*p)++;
                while (**p && **p != '"') {
                    if (**p == '\\')
                        (*p)++;
                    if (**p)
                        (*p)++;
                }
                if (**p == '"')
                    (*p)++;
            } else if (**p == '[') {
                int depth = 1;
                (*p)++;
                while (**p && depth > 0) {
                    if (**p == '[')
                        depth++;
                    else if (**p == ']')
                        depth--;
                    (*p)++;
                }
            } else if (**p == '{') {
                int depth = 1;
                (*p)++;
                while (**p && depth > 0) {
                    if (**p == '{')
                        depth++;
                    else if (**p == '}')
                        depth--;
                    (*p)++;
                }
            } else {
                while (**p && **p != ',' && **p != '}')
                    (*p)++;
            }
        }
    }
    if (**p != '}')
        return -1;
    (*p)++;
    return 0;
}

static int parse_header(safetensors_file_t *sf) {
    const char *p = sf->header_json;
    skip_whitespace(&p);
    if (*p != '{')
        return -1;
    p++;

    sf->num_tensors = 0;

    while (*p && *p != '}') {
        skip_whitespace(&p);
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p == '}')
            break;

        char name[256];
        if (parse_string(&p, name, sizeof(name)) != 0)
            return -1;
        skip_whitespace(&p);
        if (*p != ':')
            return -1;
        p++;

        if (strcmp(name, "__metadata__") == 0) {
            skip_whitespace(&p);
            if (*p == '{') {
                int depth = 1;
                p++;
                while (*p && depth > 0) {
                    if (*p == '{')
                        depth++;
                    else if (*p == '}')
                        depth--;
                    p++;
                }
            }
            continue;
        }

        if (sf->num_tensors >= SAFETENSORS_MAX_TENSORS) {
            fprintf(stderr, "safetensors: too many tensors in %s (max %d)\n",
                    sf->path ? sf->path : "<unknown>", SAFETENSORS_MAX_TENSORS);
            return -1;
        }

        safetensor_t *t = &sf->tensors[sf->num_tensors];
        snprintf(t->name, sizeof(t->name), "%s", name);
        if (parse_tensor_entry(&p, t) != 0)
            return -1;
        sf->num_tensors++;
    }
    if (*p != '}')
        return -1;
    return 0;
}

/* ========================================================================
 * Single file operations
 * ======================================================================== */

safetensors_file_t *safetensors_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }

    size_t file_size = (size_t)st.st_size;
    if (file_size < 8) {
        close(fd);
        return NULL;
    }

    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED)
        return NULL;

    uint64_t header_size = 0;
    memcpy(&header_size, data, 8);
    if (header_size > file_size - 8) {
        munmap(data, file_size);
        return NULL;
    }

    safetensors_file_t *sf = calloc(1, sizeof(safetensors_file_t));
    if (!sf) {
        munmap(data, file_size);
        return NULL;
    }

    sf->path = strdup(path);
    sf->data = data;
    sf->file_size = file_size;
    sf->header_size = (size_t)header_size;

    sf->header_json = malloc(header_size + 1);
    if (!sf->header_json) {
        safetensors_close(sf);
        return NULL;
    }
    memcpy(sf->header_json, (char *)data + 8, header_size);
    sf->header_json[header_size] = '\0';

    if (parse_header(sf) != 0) {
        safetensors_close(sf);
        return NULL;
    }

    return sf;
}

void safetensors_close(safetensors_file_t *sf) {
    if (!sf)
        return;
    if (sf->data)
        munmap(sf->data, sf->file_size);
    free(sf->path);
    free(sf->header_json);
    free(sf);
}

const void *safetensors_data(const safetensors_file_t *sf, const safetensor_t *t) {
    return (const char *)sf->data + 8 + sf->header_size + t->data_offset;
}

int64_t safetensor_numel(const safetensor_t *t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++)
        n *= t->shape[i];
    return n;
}

static float bf16_to_f32(uint16_t bf16) {
    uint32_t f32 = ((uint32_t)bf16) << 16;
    float result;
    memcpy(&result, &f32, sizeof(float));
    return result;
}

float *safetensors_get_f32(const safetensors_file_t *sf, const safetensor_t *t) {
    int64_t n = safetensor_numel(t);
    if (n <= 0)
        return NULL;

    float *out = malloc(n * sizeof(float));
    if (!out)
        return NULL;

    const void *data = safetensors_data(sf, t);

    switch (t->dtype) {
    case DTYPE_F32:
        memcpy(out, data, n * sizeof(float));
        break;
    case DTYPE_BF16: {
        const uint16_t *src = (const uint16_t *)data;
        for (int64_t i = 0; i < n; i++)
            out[i] = bf16_to_f32(src[i]);
        break;
    }
    default:
        free(out);
        return NULL;
    }
    return out;
}

/* ========================================================================
 * Multi-shard operations
 * ======================================================================== */

static int cmp_shard_names(const void *a, const void *b) {
    const char *sa = (const char *)a;
    const char *sb = (const char *)b;
    return strcmp(sa, sb);
}

static int parse_shard_name(const char *name, int *idx, int *total) {
    int i = 0, n = 0, consumed = 0;
    if (sscanf(name, "model-%d-of-%d.safetensors%n", &i, &n, &consumed) != 2)
        return -1;
    if (name[consumed] != '\0' || i <= 0 || n <= 0 || i > n)
        return -1;
    *idx = i;
    *total = n;
    return 0;
}

multi_safetensors_t *multi_safetensors_open(const char *model_dir) {
    multi_safetensors_t *ms = calloc(1, sizeof(multi_safetensors_t));
    if (!ms)
        return NULL;

    char path[4096];

    /* Try single file first */
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    safetensors_file_t *sf = safetensors_open(path);
    if (sf) {
        ms->shards[0] = sf;
        ms->num_shards = 1;
        return ms;
    }

    /* Scan directory for shard files */
    DIR *dir = opendir(model_dir);
    if (!dir) {
        free(ms);
        return NULL;
    }

    struct dirent *entry;
    char shard_names[SAFETENSORS_MAX_SHARDS][256];
    int seen[SAFETENSORS_MAX_SHARDS] = {0};
    int n_shards = 0;
    int expected_total = 0;

    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "model-", 6) == 0 &&
            strstr(entry->d_name, ".safetensors") != NULL) {
            int shard_idx = 0, shard_total = 0;
            if (parse_shard_name(entry->d_name, &shard_idx, &shard_total) != 0) {
                fprintf(stderr, "multi_safetensors_open: bad shard name: %s\n", entry->d_name);
                closedir(dir);
                free(ms);
                return NULL;
            }
            if (shard_total > SAFETENSORS_MAX_SHARDS) {
                fprintf(stderr, "multi_safetensors_open: too many shards (%d > %d)\n", shard_total,
                        SAFETENSORS_MAX_SHARDS);
                closedir(dir);
                free(ms);
                return NULL;
            }
            if (expected_total == 0) {
                expected_total = shard_total;
            } else if (expected_total != shard_total) {
                fprintf(stderr, "multi_safetensors_open: inconsistent shard total in %s\n",
                        entry->d_name);
                closedir(dir);
                free(ms);
                return NULL;
            }
            if (seen[shard_idx - 1]) {
                fprintf(stderr, "multi_safetensors_open: duplicate shard index %d\n", shard_idx);
                closedir(dir);
                free(ms);
                return NULL;
            }
            seen[shard_idx - 1] = 1;

            snprintf(shard_names[n_shards], sizeof(shard_names[n_shards]), "%s", entry->d_name);
            n_shards++;
        }
    }
    closedir(dir);

    if (n_shards == 0) {
        fprintf(stderr, "multi_safetensors_open: no safetensors files in %s\n", model_dir);
        free(ms);
        return NULL;
    }
    if (expected_total != n_shards) {
        fprintf(stderr, "multi_safetensors_open: expected %d shard(s), found %d in %s\n",
                expected_total, n_shards, model_dir);
        free(ms);
        return NULL;
    }
    for (int i = 0; i < expected_total; i++) {
        if (!seen[i]) {
            fprintf(stderr, "multi_safetensors_open: missing shard %d of %d in %s\n", i + 1,
                    expected_total, model_dir);
            free(ms);
            return NULL;
        }
    }

    /* Sort shard names to ensure consistent ordering */
    qsort(shard_names, n_shards, sizeof(shard_names[0]), cmp_shard_names);

    /* Open each shard */
    for (int i = 0; i < n_shards; i++) {
        snprintf(path, sizeof(path), "%s/%s", model_dir, shard_names[i]);
        ms->shards[i] = safetensors_open(path);
        if (!ms->shards[i]) {
            fprintf(stderr, "multi_safetensors_open: failed to open %s\n", path);
            multi_safetensors_close(ms);
            return NULL;
        }
    }
    ms->num_shards = n_shards;
    return ms;
}

void multi_safetensors_close(multi_safetensors_t *ms) {
    if (!ms)
        return;
    for (int i = 0; i < ms->num_shards; i++) {
        safetensors_close(ms->shards[i]);
    }
    free(ms);
}

int multi_safetensors_data_nbytes(const multi_safetensors_t *ms, size_t *out_nbytes) {
    if (!ms || !out_nbytes)
        return -1;
    size_t total = 0;
    for (int s = 0; s < ms->num_shards; s++) {
        const safetensors_file_t *sf = ms->shards[s];
        for (int i = 0; i < sf->num_tensors; i++) {
            size_t n = sf->tensors[i].data_size;
            if (n > SIZE_MAX - total)
                return -1;
            total += n;
        }
    }
    *out_nbytes = total;
    return 0;
}

const safetensor_t *multi_safetensors_find(const multi_safetensors_t *ms,
                                           const char *name,
                                           safetensors_file_t **out_sf) {
    for (int s = 0; s < ms->num_shards; s++) {
        safetensors_file_t *sf = ms->shards[s];
        for (int i = 0; i < sf->num_tensors; i++) {
            if (strcmp(sf->tensors[i].name, name) == 0) {
                if (out_sf)
                    *out_sf = sf;
                return &sf->tensors[i];
            }
        }
    }
    return NULL;
}

const char *multi_safetensors_weight_prefix(const multi_safetensors_t *ms,
                                            const char *tensor_name) {
    if (multi_safetensors_find(ms, tensor_name, NULL))
        return "";

    char prefixed[sizeof(((safetensor_t *)0)->name)];
    int n = snprintf(prefixed, sizeof(prefixed), "model.%s", tensor_name);
    if (n < 0 || (size_t)n >= sizeof(prefixed))
        return NULL;
    return multi_safetensors_find(ms, prefixed, NULL) ? "model." : NULL;
}
