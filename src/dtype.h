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

#endif /* EMBED_DTYPE_H */
