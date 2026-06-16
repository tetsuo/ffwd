#ifndef EMBED_DTYPE_H
#define EMBED_DTYPE_H

#include <stdint.h>
#include <string.h>

/* BF16 is the upper 16 bits of an IEEE-754 binary32, so widening one value is a
 * left shift reinterpreted as a float. Shared by every host-side caller (the
 * CPU loader, the safetensors reader); the CUDA backend keeps its own
 * __nv_bfloat16 device kernels, and kernels.c keeps the SIMD buffer variant
 * embed_bf16_to_f32_buf for hot, contiguous conversions. */
static inline float embed_bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* IEEE-754 binary16 -> binary32. Unlike BF16, F16 has a 5-bit exponent and a
 * 10-bit mantissa, so widening is a field re-pack, not a shift: rebias the
 * exponent (15 -> 127), shift the mantissa up by 13, and special-case zeros,
 * subnormals, and inf/NaN. Used by the loaders to accept F16 safetensors
 * snapshots (e.g. the public multilingual-e5 weights); there is no fused F16
 * kernel, so weights are widened to F32 once at load. */
static inline float embed_f16_to_f32(uint16_t f16) {
    uint32_t sign = (uint32_t)(f16 & 0x8000u) << 16;
    uint32_t exp = (f16 >> 10) & 0x1fu;
    uint32_t mant = f16 & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign; /* signed zero */
        } else {
            /* Subnormal half: renormalize into a normal float. Shift the
             * mantissa left until its leading 1 reaches the hidden-bit
             * position, decrementing the exponent for each shift. */
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13); /* inf or NaN */
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

#endif /* EMBED_DTYPE_H */
