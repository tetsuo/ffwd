#include "ffwd.h"

#include <math.h>

/* Vector helpers */

int ffwd_l2_normalize(float *vec, int dim) {
    if (!vec || dim <= 0)
        return -1;

    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++)
        norm_sq += vec[i] * vec[i];

    if (!(norm_sq > 0.0f) || !isfinite(norm_sq))
        return -1;

    float inv_norm = 1.0f / sqrtf(norm_sq);
    for (int i = 0; i < dim; i++)
        vec[i] *= inv_norm;
    return 0;
}

float ffwd_cosine_similarity(const float *a, const float *b, int dim) {
    if (!a || !b || dim <= 0)
        return 0.0f;

    float dot = 0.0f;
    float norm_a_sq = 0.0f;
    float norm_b_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        float av = a[i];
        float bv = b[i];
        dot += av * bv;
        norm_a_sq += av * av;
        norm_b_sq += bv * bv;
    }

    if (!(norm_a_sq > 0.0f) || !(norm_b_sq > 0.0f) || !isfinite(norm_a_sq) || !isfinite(norm_b_sq) ||
        !isfinite(dot))
        return 0.0f;

    double denom = sqrt((double)norm_a_sq) * sqrt((double)norm_b_sq);
    if (!(denom > 0.0) || !isfinite(denom))
        return 0.0f;
    return (float)((double)dot / denom);
}
