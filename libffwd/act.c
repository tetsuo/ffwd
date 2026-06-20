/*
 * act.c - activations
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "impl.h"
#include "kernels.h"

#if (defined(__AVX512F__) || defined(__AVX2__)) && \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#    include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#    include <arm_neon.h>
#endif

#ifdef USE_BLAS
#    ifdef __APPLE__
#        include <Accelerate/Accelerate.h>
#    else
#        include <cblas.h>
#    endif
#endif

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

/* Activation kernels: softmax, SiLU-gated multiply, and the erf and tanh GeLU
 * variants. */

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void silu_mul_inplace(float *gate, const float *up, int n) {
#if defined(__APPLE__) && defined(USE_BLAS)
    enum { CHUNK = 4096 };
    float exp_neg[CHUNK];

    for (int base = 0; base < n; base += CHUNK) {
        int len = n - base;
        if (len > CHUNK)
            len = CHUNK;

        for (int i = 0; i < len; i++)
            exp_neg[i] = -gate[base + i];
        vvexpf(exp_neg, exp_neg, &len);

        for (int i = 0; i < len; i++) {
            float g = gate[base + i];
            gate[base + i] = (g / (1.0f + exp_neg[i])) * up[base + i];
        }
    }
#else
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
#endif
}

void gelu_inplace(float *x, int n) {
    /* Exact erf GeLU; 0.70710678... = 1/sqrt(2). */
    for (int i = 0; i < n; i++)
        x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678118654752f));
}

void gelu_tanh_inplace(float *x, int n) {
    /* Tanh-approximation GeLU; 0.79788456... = sqrt(2/pi). */
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float inner = 0.79788456080286536f * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

void softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val)
                max_val = row[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}
