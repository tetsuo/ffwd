extern "C" {
#include "cuda.h"
#include "model_types.h"
}

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    CUDA_DTYPE_F32 = 0,
    CUDA_DTYPE_BF16 = 1,
    CUDA_DTYPE_F16 = 2,
} cuda_dtype_t;

typedef struct {
    void *d; // device buffer; element type is described by dtype
    int rows;
    int cols;
    cuda_dtype_t dtype;
    int bf16; // compatibility shorthand for dtype == CUDA_DTYPE_BF16
    int f16;  // compatibility shorthand for dtype == CUDA_DTYPE_F16
} cuda_matrix_t;

typedef struct {
    cuda_matrix_t qkv, wo;
    float *qkv_bias;
    float *q_norm;
    float *k_norm;
    float *input_norm;
    float *post_attn_norm;
    cuda_matrix_t gate_up_proj, down_proj;
    /* BERT family (NULL for Qwen3). The two block LayerNorm weights reuse
     * input_norm (post-attention) and post_attn_norm (post-FFN); gate_up_proj
     * holds the single BERT up_proj. These add the biases those slots lack. */
    float *o_bias;
    float *attn_ln_bias;
    float *ffn_inter_bias;
    float *ffn_out_bias;
    float *ffn_ln_bias;
} cuda_layer_t;

struct ffwd_cuda_ctx {
    ffwd_model_t *cpu;
    int own_cpu; // free ctx->cpu in ffwd_cuda_free only when this context owns it
    ffwd_config_t config;
    cuda_dtype_t weight_dtype;
    int weights_bf16;
    int weights_f16;
    cudaStream_t stream;
    cublasHandle_t blas;
    cublasLtHandle_t lt; // cuBLASLt handle for fused bias/GeLU epilogue GEMMs
    void *lt_workspace;  // device workspace for cublasLtMatmul
    size_t lt_ws_bytes;
    void *lt_bias16; // 16-bit cast of an epilogue bias (cuBLASLt rejects F32 bias on 16-bit out)
    // Cached cuBLASLt plan for the repeated gate-up shape (rebuilt only when the
    // shape changes, e.g. a new total-token count); skips per-layer heuristics.
    cublasLtMatmulDesc_t lt_desc;
    cublasLtMatrixLayout_t lt_A, lt_B, lt_D;
    cublasLtMatmulHeuristicResult_t lt_heur;
    int lt_have, lt_k_rows, lt_k_in, lt_k_out, lt_k_xs, lt_k_epi, lt_k_io, lt_k_od;
    cuda_matrix_t embed_tokens;
    cuda_layer_t *layers;
    float *norm;
    /* BERT family (NULL for Qwen3): learned absolute position embeddings, the
     * token-type[0] embedding (row 0 only, [hidden]), and the embedding
     * LayerNorm weight/bias. */
    float *position_embeddings;
    float *token_type_embedding;
    float *ffwd_ln_w;
    float *ffwd_ln_b;

    int seq_cap;
    int scratch_key;
    int batch_cap;
    int max_seq_cap;
    float *x;
    float *x_norm;
    float *qkv;
    void *qkv_16; // BERT reduced-precision QKV projection (flash path)
    float *attn_out;
    float *ffn_gate_up;
    void *ffn_gate_up_16; // BERT reduced-precision FFN intermediate
    void *resid_delta_16; // BERT reduced-precision residual delta (wo/down output)
    void *x_bf16;         // Persistent BERT BF16 residual stream (gated)
    void *x_f16;          // Persistent BERT FP16 residual stream (explicit f16 path)
    void *act_bf16;       // BF16 cast of a GEMM activation operand (bf16 weights)
    void *act_f16;        // FP16 cast of a GEMM activation operand (f16 weights)
    float *weight_f32;    // F32 widen scratch for one BF16 weight (memory-only bf16)
    size_t weight_f32_elems;
    float *pooled_out;
    float *rope_cos;
    float *rope_sin;
    int *token_ids;
    int *offsets;
    int *offsets_host;
    int *positions;
    float *kexp;
    void *kexp_bf16; // BF16 K expansion for tensor-core streaming attention
    void *kexp_f16;  // FP16 K expansion for tensor-core streaming attention
    float *vexp;
    float *attn_scores;
    void *attn_probs; // BF16 softmax output (bf16 weights)
    long long attn_scores_elems;
    long long attn_probs_elems;
    // Equal-length micro-batch attention: pointer arrays for batched GEMMs,
    // built once per forward pass (layer-invariant buffer addresses).
    // attn_G is the number of sequences whose score tensors fit the scratch
    // budget at once; 0 disables the batched path for this forward.
    const void **attn_ptrs; // device: [K|Q|S|V|P|O] x batch x n_heads
    const void **attn_ptrs_host;
    int attn_ptrs_cap; // capacity in pointer entries
    int attn_G;
    int attn_L;
    int *span_starts;
    int *span_lens;
    int pooled_rows_cap;
    int span_cap;
};

// Scratch budget for the batched-attention score tensors (F32 elements).
enum { CUDA_ATTN_SCORES_BUDGET = 64 * 1024 * 1024 };
// Above this sequence length the per-sequence strided-batched loop beats the
// pointer-array batched GEMMs (L4: batched wins <=80tok, ties ~140, loses
// >=256; both paths are identical precision).
enum { CUDA_ATTN_BATCHED_MAX_LEN = 192 };

#define CUDA_CHECK(expr)                                                             \
    do {                                                                             \
        cudaError_t _e = (expr);                                                     \
        if (_e != cudaSuccess) {                                                     \
            fprintf(stderr, "cuda: %s failed: %s\n", #expr, cudaGetErrorString(_e)); \
            return -1;                                                               \
        }                                                                            \
    } while (0)

#define CUBLAS_CHECK(expr)                                                          \
    do {                                                                            \
        cublasStatus_t _s = (expr);                                                 \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                          \
            fprintf(stderr, "cuda: %s failed: cublas status %d\n", #expr, (int)_s); \
            return -1;                                                              \
        }                                                                           \
    } while (0)

static int parse_gemm_compute(const char *mode, cublasComputeType_t *out) {
    if (!mode || !mode[0] || !strcmp(mode, "f32")) {
        *out = CUBLAS_COMPUTE_32F;
        return 0;
    }
    if (!strcmp(mode, "tf32")) {
        *out = CUBLAS_COMPUTE_32F_FAST_TF32;
        return 0;
    }
    if (!strcmp(mode, "bf16")) {
        *out = CUBLAS_COMPUTE_32F_FAST_16BF;
        return 0;
    }
    if (!strcmp(mode, "16f")) {
        *out = CUBLAS_COMPUTE_32F_FAST_16F;
        return 0;
    }
    return -1;
}

static cublasComputeType_t g_gemm_compute = CUBLAS_COMPUTE_32F;
static int g_gemm_compute_set = 0;

int ffwd_cuda_set_fast_gemm(const char *mode) {
    cublasComputeType_t ct;
    if (parse_gemm_compute(mode, &ct) != 0)
        return -1;
    g_gemm_compute = ct;
    g_gemm_compute_set = 1;
    return 0;
}

static void cuda_matrix_free(cuda_matrix_t *m) {
    if (!m)
        return;
    cudaFree(m->d);
    memset(m, 0, sizeof(*m));
}

static float bf16_to_f32(uint16_t v) {
    uint32_t u = ((uint32_t)v) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// Round F32 to BF16 (truncate the low 16 bits, round to nearest even). NaNs are
// preserved. BF16 shares F32's 8-bit exponent, so this never overflows.
static uint16_t f32_to_bf16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    if ((x & 0x7fffffffu) > 0x7f800000u) // NaN
        return (uint16_t)((x >> 16) | 0x0040u);
    x += 0x7fffu + ((x >> 16) & 1u); // round to nearest even
    return (uint16_t)(x >> 16);
}

static uint16_t f32_to_f16(float f) {
    __half h = __float2half_rn(f);
    uint16_t out;
    memcpy(&out, &h, sizeof(out));
    return out;
}

// Explicit GPU weight-storage override. -1 means "follow the model file dtype"
// at context creation time, matching the CPU and MLX backends. The resolved
// choice is stored on ffwd_cuda_ctx_t; inference code must not read this global,
// or mixed-dtype multi-model servers inherit whichever model loaded last.
static int g_weight_dtype_override = -1;

int ffwd_cuda_set_weights_bf16(int on) {
    g_weight_dtype_override = on ? CUDA_DTYPE_BF16 : CUDA_DTYPE_F32;
    return 0;
}

int ffwd_cuda_set_weights_f16(int on) {
    g_weight_dtype_override = on ? CUDA_DTYPE_F16 : CUDA_DTYPE_F32;
    return 0;
}

static int copy_weight_host_f32(float *dst, const ffwd_weight_ref_t *w, size_t count) {
    if (!dst || !w || !w->data)
        return -1;
    if (w->dtype == DTYPE_F32) {
        memcpy(dst, w->data, count * sizeof(float));
        return 0;
    }
    if (w->dtype == DTYPE_BF16) {
        const uint16_t *src = (const uint16_t *)w->data;
        for (size_t i = 0; i < count; i++)
            dst[i] = bf16_to_f32(src[i]);
        return 0;
    }
    return -1;
}

// Upload a host F32 weight buffer to the device, rounding to the requested
// storage dtype when reduced precision is selected. Sets m->d, rows, cols, dtype.
static int upload_weight(
    cuda_matrix_t *m, const float *src, size_t count, int rows, int cols, cuda_dtype_t dtype) {
    cudaError_t e;
    if (dtype == CUDA_DTYPE_BF16 || dtype == CUDA_DTYPE_F16) {
        uint16_t *tmp16 = (uint16_t *)malloc(count * sizeof(uint16_t));
        if (!tmp16)
            return -1;
        for (size_t i = 0; i < count; i++) {
            tmp16[i] = dtype == CUDA_DTYPE_BF16 ? f32_to_bf16(src[i]) : f32_to_f16(src[i]);
        }
        e = cudaMalloc(&m->d, count * sizeof(uint16_t));
        if (e != cudaSuccess) {
            fprintf(stderr, "cudaMalloc weight failed: %s\n", cudaGetErrorString(e));
            free(tmp16);
            return -1;
        }
        e = cudaMemcpy(m->d, tmp16, count * sizeof(uint16_t), cudaMemcpyHostToDevice);
        free(tmp16);
    } else {
        e = cudaMalloc(&m->d, count * sizeof(float));
        if (e != cudaSuccess) {
            fprintf(stderr, "cudaMalloc weight failed: %s\n", cudaGetErrorString(e));
            return -1;
        }
        e = cudaMemcpy(m->d, src, count * sizeof(float), cudaMemcpyHostToDevice);
    }
    if (e != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy weight failed: %s\n", cudaGetErrorString(e));
        cuda_matrix_free(m);
        return -1;
    }
    m->rows = rows;
    m->cols = cols;
    m->dtype = dtype;
    m->bf16 = dtype == CUDA_DTYPE_BF16;
    m->f16 = dtype == CUDA_DTYPE_F16;
    return 0;
}

static int load_matrix(
    cuda_matrix_t *m, const ffwd_weight_ref_t *w, int rows, int cols, cuda_dtype_t weight_dtype) {
    if (!m || !w || rows <= 0 || cols <= 0)
        return -1;
    size_t count = (size_t)rows * (size_t)cols;
    float *tmp = (float *)malloc(count * sizeof(float));
    if (!tmp)
        return -1;
    if (copy_weight_host_f32(tmp, w, count) != 0) {
        free(tmp);
        return -1;
    }
    int r = upload_weight(m, tmp, count, rows, cols, weight_dtype);
    free(tmp);
    return r;
}

static int load_qkv_matrix(cuda_matrix_t *m,
                           const ffwd_layer_t *src,
                           const ffwd_config_t *c,
                           cuda_dtype_t weight_dtype) {
    if (!m || !src || !c)
        return -1;
    int rows = c->q_dim + 2 * c->kv_dim;
    int cols = c->hidden_size;
    size_t q_count = (size_t)c->q_dim * cols;
    size_t kv_count = (size_t)c->kv_dim * cols;
    size_t total = q_count + 2 * kv_count;
    float *tmp = (float *)malloc(total * sizeof(float));
    if (!tmp)
        return -1;
    if (copy_weight_host_f32(tmp, &src->wq, q_count) != 0 ||
        copy_weight_host_f32(tmp + q_count, &src->wk, kv_count) != 0 ||
        copy_weight_host_f32(tmp + q_count + kv_count, &src->wv, kv_count) != 0) {
        free(tmp);
        return -1;
    }
    int r = upload_weight(m, tmp, total, rows, cols, weight_dtype);
    free(tmp);
    return r;
}

static int load_gate_up_matrix(cuda_matrix_t *m,
                               const ffwd_layer_t *src,
                               const ffwd_config_t *c,
                               cuda_dtype_t weight_dtype) {
    if (!m || !src || !c)
        return -1;
    int rows = 2 * c->intermediate_size;
    int cols = c->hidden_size;
    size_t proj_count = (size_t)c->intermediate_size * cols;
    size_t total = 2 * proj_count;
    float *tmp = (float *)malloc(total * sizeof(float));
    if (!tmp)
        return -1;
    if (copy_weight_host_f32(tmp, &src->gate_proj, proj_count) != 0 ||
        copy_weight_host_f32(tmp + proj_count, &src->up_proj, proj_count) != 0) {
        free(tmp);
        return -1;
    }
    int r = upload_weight(m, tmp, total, rows, cols, weight_dtype);
    free(tmp);
    return r;
}

static int load_vector(float **out, const float *src, int n) {
    if (!out || !src || n <= 0)
        return -1;
    cudaError_t e = cudaMalloc((void **)out, (size_t)n * sizeof(float));
    if (e != cudaSuccess) {
        fprintf(stderr, "cudaMalloc vector failed: %s\n", cudaGetErrorString(e));
        return -1;
    }
    e = cudaMemcpy(*out, src, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy vector failed: %s\n", cudaGetErrorString(e));
        cudaFree(*out);
        *out = NULL;
        return -1;
    }
    return 0;
}

static int load_qkv_bias(float **out, const ffwd_layer_t *src, const ffwd_config_t *c) {
    int n = c->q_dim + 2 * c->kv_dim;
    float *host = (float *)malloc((size_t)n * sizeof(float));
    if (!host)
        return -1;
    memcpy(host, src->q_bias, (size_t)c->q_dim * sizeof(float));
    memcpy(host + c->q_dim, src->k_bias, (size_t)c->kv_dim * sizeof(float));
    memcpy(host + c->q_dim + c->kv_dim, src->v_bias, (size_t)c->kv_dim * sizeof(float));
    int rc = load_vector(out, host, n);
    free(host);
    return rc;
}

__global__ static void add_row_bias_kernel(float *x, const float *bias, int rows, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = rows * dim;
    if (idx < count)
        x[idx] += bias[idx % dim];
}

__global__ static void ffwd_lookup_kernel(
    float *x, const int *ids, const void *emb, int emb_dtype, int total, int hidden, int vocab) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = total * hidden;
    if (idx >= count)
        return;
    int tok = idx / hidden;
    int d = idx - tok * hidden;
    int id = ids[tok];
    if (id < 0 || id >= vocab) {
        x[idx] = 0.0f;
        return;
    }
    size_t off = (size_t)id * hidden + d;
    if (emb_dtype == CUDA_DTYPE_BF16)
        x[idx] = __bfloat162float(((const __nv_bfloat16 *)emb)[off]);
    else if (emb_dtype == CUDA_DTYPE_F16)
        x[idx] = __half2float(((const __half *)emb)[off]);
    else
        x[idx] = ((const float *)emb)[off];
}

// Cast n F32 values to BF16 (used to feed BF16-weight projection GEMMs).
__global__ static void cast_f32_to_bf16_kernel(__nv_bfloat16 *out, const float *in, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2bfloat16(in[i]);
}

// Widen n BF16 values to F32 (used to run an exact F32 GEMM on BF16-stored
// weights - memory-only BF16, no precision loss from the operands).
__global__ static void cast_bf16_to_f32_kernel(float *out, const __nv_bfloat16 *in, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __bfloat162float(in[i]);
}

// Cast n F32 values to FP16 (used to feed FP16-weight projection GEMMs).
__global__ static void cast_f32_to_f16_kernel(__half *out, const float *in, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half(in[i]);
}

// Widen n FP16 values to F32 (used by exact/TF32 compute with FP16-stored
// weights, and to expose the final residual stream to existing F32 pooling).
__global__ static void cast_f16_to_f32_kernel(float *out, const __half *in, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __half2float(in[i]);
}

__device__ static float block_sum(float v) {
    __shared__ float partial[8];
    __shared__ float total;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;

    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    if (lane == 0)
        partial[warp] = v;
    __syncthreads();

    if (warp == 0) {
        v = lane < warps ? partial[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffff, v, offset);
        if (lane == 0)
            total = v;
    }
    __syncthreads();
    return total;
}

__device__ static float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return __shfl_sync(0xffffffff, v, 0);
}

__device__ static inline void store_act(float *out, size_t i, float v) { out[i] = v; }

__device__ static inline void store_act(__nv_bfloat16 *out, size_t i, float v) {
    out[i] = __float2bfloat16(v);
}

__device__ static inline void store_act(__half *out, size_t i, float v) { out[i] = __float2half(v); }

__device__ static inline float load_act(const float *p, size_t i) { return p[i]; }

__device__ static inline float load_act(const __nv_bfloat16 *p, size_t i) {
    return __bfloat162float(p[i]);
}

__device__ static inline float load_act(const __half *p, size_t i) { return __half2float(p[i]); }

__device__ static inline float load_embedding_value(const void *emb, int emb_dtype, size_t i) {
    if (emb_dtype == CUDA_DTYPE_BF16)
        return __bfloat162float(((const __nv_bfloat16 *)emb)[i]);
    if (emb_dtype == CUDA_DTYPE_F16)
        return __half2float(((const __half *)emb)[i]);
    return ((const float *)emb)[i];
}

// OUT_T is float, or __nv_bfloat16 when the result feeds a BF16-weight
// projection GEMM directly (same rounding as a separate cast kernel).
template <typename OUT_T>
__global__ static void
rms_norm_kernel(OUT_T *out, const float *x, const float *weight, int rows, int dim, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = x[(size_t)row * dim + d];
        sum += v * v;
    }
    float total = block_sum(sum);
    float inv = rsqrtf(total / (float)dim + eps);
    for (int d = tid; d < dim; d += blockDim.x)
        store_act(out, (size_t)row * dim + d, x[(size_t)row * dim + d] * inv * weight[d]);
}

// One grid covers Q and K: blockIdx.y < n_heads selects the Q head with
// q_norm/q_offset, the remaining blocks the K head with k_norm/k_offset.
__global__ static void rms_norm_rope_qk_kernel(float *x,
                                               const float *q_norm,
                                               const float *k_norm,
                                               const int *positions,
                                               const float *cosv,
                                               const float *sinv,
                                               int rows,
                                               int n_heads,
                                               int n_kv_heads,
                                               int head_dim,
                                               int row_stride,
                                               int q_offset,
                                               int k_offset,
                                               int qk_norm,
                                               float eps) {
    __shared__ float partial[4];
    __shared__ float total;
    int row = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int half = head_dim / 2;
    int pair_d = tid < half ? tid + half : tid - half;
    float sign = tid < half ? -1.0f : 1.0f;
    const float *weight = head < n_heads ? q_norm : k_norm;
    int base_offset = head < n_heads ? q_offset : k_offset;
    if (head >= n_heads)
        head -= n_heads;
    size_t base = (size_t)row * row_stride + base_offset + head * head_dim;
    float v = tid < head_dim ? x[base + tid] : 0.0f;
    float pair_v = tid < head_dim ? x[base + pair_d] : 0.0f;
    if (qk_norm) {
        float sum = v * v;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0)
            partial[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            sum = lane < 4 ? partial[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            if (lane == 0)
                total = sum;
        }
        __syncthreads();
    }

    if (tid < head_dim) {
        int pos = positions[row];
        float c = cosv[(size_t)pos * head_dim + tid];
        float s = sinv[(size_t)pos * head_dim + tid];
        float a = v;
        float b = pair_v;
        if (qk_norm) {
            float inv = rsqrtf(total / (float)head_dim + eps);
            a *= inv * weight[tid];
            b *= inv * weight[pair_d];
        }
        x[base + tid] = a * c + sign * b * s;
    }
}

// Writes SiLU(gate)*up to out with out_stride. The F32 path writes in place
// over the gate half (out = gate_up, out_stride = 2*intermediate); the BF16
// path writes a packed [rows x intermediate] BF16 matrix for the down GEMM.
template <typename OUT_T>
__global__ static void
silu_mul_packed_kernel(OUT_T *out, int out_stride, const float *gate_up, int rows, int intermediate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * intermediate;
    if (i < n) {
        int row = i / intermediate;
        int d = i - row * intermediate;
        size_t base = (size_t)row * (2 * intermediate);
        float g = gate_up[base + d];
        float u = gate_up[base + intermediate + d];
        store_act(out, (size_t)row * out_stride + d, (g / (1.0f + expf(-g))) * u);
    }
}

// Expand K and V from the n_kv_heads layout to a contiguous [total x q_dim]
// per-query-head layout (GQA: each kv head repeated n_heads/n_kv_heads times)
// so the attention GEMMs can stride uniformly over query heads.
// K_T/V_T are float for materialized F32/cublas attention, or __nv_bfloat16 when
// a tensor-core/BF16 path consumes the expanded tile directly.
template <typename K_T, typename V_T, typename QKV_T>
__global__ static void attn_expand_kv_kernel(K_T *kexp,
                                             V_T *vexp,
                                             const QKV_T *qkv,
                                             int total,
                                             int n_heads,
                                             int n_kv_heads,
                                             int head_dim,
                                             int qkv_dim,
                                             int k_offset,
                                             int v_offset) {
    int q_dim = n_heads * head_dim;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)total * q_dim;
    if (idx >= count)
        return;
    int t = (int)(idx / q_dim);
    int rem = (int)(idx - (size_t)t * q_dim);
    int h = rem / head_dim;
    int e = rem - h * head_dim;
    int kv = h / (n_heads / n_kv_heads);
    size_t base = (size_t)t * qkv_dim;
    store_act(kexp, idx, load_act(qkv, base + k_offset + kv * head_dim + e));
    store_act(vexp, idx, load_act(qkv, base + v_offset + kv * head_dim + e));
}

// Row softmax over the key dimension for one sequence's score tensor laid out
// as scores[head * L * L + i * L + j]; normalizes across j for each (head, i).
// One block per (head, query) row; scores are pre-scaled by the QK^T GEMM.
// P_T is float with probs == scores (in place), or __nv_bfloat16 when the
// probabilities feed a BF16 @V GEMM.
template <typename P_T> __global__ static void attn_softmax_kernel(float *scores, P_T *probs, int L) {
    float *s = scores + (size_t)blockIdx.x * L;
    P_T *p = probs + (size_t)blockIdx.x * L;
    int tid = threadIdx.x;
    __shared__ float red[256];

    float m = -3.402823466e+38F;
    for (int j = tid; j < L; j += blockDim.x)
        m = fmaxf(m, s[j]);
    red[tid] = m;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (tid < o)
            red[tid] = fmaxf(red[tid], red[tid + o]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    float sum = 0.0f;
    for (int j = tid; j < L; j += blockDim.x) {
        float e = __expf(s[j] - m);
        s[j] = e;
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (tid < o)
            red[tid] += red[tid + o];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    for (int j = tid; j < L; j += blockDim.x)
        store_act(p, (size_t)j, s[j] * inv);
}

template <typename P_T>
__global__ static void attn_causal_softmax_kernel(float *scores, P_T *probs, int L) {
    float *s = scores + (size_t)blockIdx.x * L;
    P_T *p = probs + (size_t)blockIdx.x * L;
    int query = blockIdx.x % L;
    int keys = query + 1;
    int tid = threadIdx.x;
    __shared__ float red[256];

    float m = -3.402823466e+38F;
    for (int j = tid; j < keys; j += blockDim.x)
        m = fmaxf(m, s[j]);
    red[tid] = m;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (tid < o)
            red[tid] = fmaxf(red[tid], red[tid + o]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    float sum = 0.0f;
    for (int j = tid; j < keys; j += blockDim.x) {
        float e = __expf(s[j] - m);
        s[j] = e;
        sum += e;
    }
    red[tid] = sum;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (tid < o)
            red[tid] += red[tid + o];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    for (int j = tid; j < L; j += blockDim.x)
        store_act(p, (size_t)j, j < keys ? s[j] * inv : 0.0f);
}

__device__ static __forceinline__ uint32_t bf16_pair_bits(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}

__device__ static __forceinline__ uint32_t bf16_pair_bits(float lo, float hi) {
    return bf16_pair_bits(__float2bfloat16(lo), __float2bfloat16(hi));
}

__device__ static __forceinline__ uint32_t f16_pair_bits(__half lo, __half hi) {
    return (uint32_t)__half_as_ushort(lo) | ((uint32_t)__half_as_ushort(hi) << 16);
}

__device__ static __forceinline__ uint32_t f16_pair_bits(float lo, float hi) {
    return f16_pair_bits(__float2half(lo), __float2half(hi));
}

__device__ static __forceinline__ float warp_group4_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}

__device__ static __forceinline__ float warp_group4_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}

__device__ static __forceinline__ void
mma_m16n16k16_bf16(float c[8], const uint32_t a[4], const uint32_t b[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
                   "f"(c[1]), "f"(c[2]), "f"(c[3]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2]), "r"(b[3]), "f"(c[4]),
                   "f"(c[5]), "f"(c[6]), "f"(c[7]));
#else
    (void)c;
    (void)a;
    (void)b;
#endif
}

__device__ static __forceinline__ void
mma_m16n16k16_f16(float c[8], const uint32_t a[4], const uint32_t b[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]),
                   "f"(c[1]), "f"(c[2]), "f"(c[3]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(c[4]), "=f"(c[5]), "=f"(c[6]), "=f"(c[7])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2]), "r"(b[3]), "f"(c[4]),
                   "f"(c[5]), "f"(c[6]), "f"(c[7]));
#else
    (void)c;
    (void)a;
    (void)b;
#endif
}

__device__ static __forceinline__ void tc_ldmatrix_x4_b16(uint32_t out[4], const void *ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t smem = (uint32_t)__cvta_generic_to_shared(ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(smem)
                 : "memory");
#else
    (void)ptr;
    out[0] = out[1] = out[2] = out[3] = 0;
#endif
}

__device__ static __forceinline__ void tc_ldmatrix_x4_trans_b16(uint32_t out[4], const void *ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t smem = (uint32_t)__cvta_generic_to_shared(ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(smem)
                 : "memory");
#else
    (void)ptr;
    out[0] = out[1] = out[2] = out[3] = 0;
#endif
}

template <int STRIDE> __device__ static __forceinline__ int tc_swz_idx(int row, int col) {
    int grain = col >> 3; // one ldmatrix row address names 8 contiguous bf16 values
    int in_grain = col & 7;
    int mask = (STRIDE == 16) ? ((row >> 2) & 1) : (STRIDE == 32) ? (row & 3) : (row & 7);
    return row * STRIDE + ((grain ^ mask) << 3) + in_grain;
}

template <int STRIDE>
__device__ static __forceinline__ void
tc_store_swz_bf16(__nv_bfloat16 *sh, int row, int col, __nv_bfloat16 v) {
    sh[tc_swz_idx<STRIDE>(row, col)] = v;
}

template <int STRIDE>
__device__ static __forceinline__ void tc_store_swz_f16(__half *sh, int row, int col, __half v) {
    sh[tc_swz_idx<STRIDE>(row, col)] = v;
}

__device__ static __forceinline__ void
tc_load_a_q_ldmatrix_swz_bf16(uint32_t a[4], const __nv_bfloat16 *qsh, int d0) {
    int lane = threadIdx.x & 31;
    int mat = lane >> 3;
    int row = ((mat & 1) << 3) + (lane & 7);
    int col = d0 + ((mat >> 1) << 3);
    tc_ldmatrix_x4_b16(a, qsh + tc_swz_idx<128>(row, col));
}

__device__ static __forceinline__ void
tc_load_b_qk_ldmatrix_swz_bf16(uint32_t b[4], const __nv_bfloat16 *ktsh, int d0) {
    int lane = threadIdx.x & 31;
    int mat = lane >> 3;
    int row = d0 + ((mat & 1) << 3) + (lane & 7);
    int col = (mat >> 1) << 3;
    tc_ldmatrix_x4_trans_b16(b, ktsh + tc_swz_idx<16>(row, col));
}

__device__ static __forceinline__ void
tc_load_b_qk_ldmatrix_swz_f16(uint32_t b[4], const __half *ktsh, int d0) {
    int lane = threadIdx.x & 31;
    int mat = lane >> 3;
    int row = d0 + ((mat & 1) << 3) + (lane & 7);
    int col = (mat >> 1) << 3;
    tc_ldmatrix_x4_trans_b16(b, ktsh + tc_swz_idx<16>(row, col));
}

__device__ static __forceinline__ void
tc_load_b_pv_bf16(uint32_t b[4], const __nv_bfloat16 *vsh, int d0) {
    int lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < 4; r++) {
        int col = d0 + ((r >> 1) << 3) + (lane >> 2);
        int key = ((r & 1) << 3) + ((lane & 3) << 1);
        b[r] = bf16_pair_bits(vsh[key * 128 + col], vsh[(key + 1) * 128 + col]);
    }
}

__device__ static __forceinline__ void tc_probs_to_a_bf16(uint32_t a[4], const float s[8]) {
#pragma unroll
    for (int r = 0; r < 4; r++)
        a[r] = bf16_pair_bits(s[2 * r], s[2 * r + 1]);
}

__device__ static __forceinline__ void tc_probs_to_a_f16(uint32_t a[4], const float s[8]) {
#pragma unroll
    for (int r = 0; r < 4; r++)
        a[r] = f16_pair_bits(s[2 * r], s[2 * r + 1]);
}

// Head-dim-parametric Q and V fragment loaders for the flash kernels. The QK S
// tile and the PV O tile are 16x16 regardless of head_dim, so only the Q-tile
// swizzle stride (HD) and the V row stride (HD) depend on it; these are the
// HD=128 tc3 loaders generalized so BERT-family flash kernels can reuse the same
// fragment layout. (The K loader tc_load_b_qk_ldmatrix_swz_bf16 is already
// HD-independent: K is staged transposed with stride KT=16.)
template <int HD>
__device__ static __forceinline__ void
tc_load_a_q_ldmatrix_swz_hd(uint32_t a[4], const __nv_bfloat16 *qsh, int d0) {
    int lane = threadIdx.x & 31;
    int mat = lane >> 3;
    int row = ((mat & 1) << 3) + (lane & 7);
    int col = d0 + ((mat >> 1) << 3);
    tc_ldmatrix_x4_b16(a, qsh + tc_swz_idx<HD>(row, col));
}

template <int HD>
__device__ static __forceinline__ void
tc_load_a_q_ldmatrix_swz_hd_f16(uint32_t a[4], const __half *qsh, int d0) {
    int lane = threadIdx.x & 31;
    int mat = lane >> 3;
    int row = ((mat & 1) << 3) + (lane & 7);
    int col = d0 + ((mat >> 1) << 3);
    tc_ldmatrix_x4_b16(a, qsh + tc_swz_idx<HD>(row, col));
}

template <int HD>
__device__ static __forceinline__ void
tc_load_b_pv_hd(uint32_t b[4], const __nv_bfloat16 *vsh, int d0) {
    int lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < 4; r++) {
        int col = d0 + ((r >> 1) << 3) + (lane >> 2);
        int key = ((r & 1) << 3) + ((lane & 3) << 1);
        b[r] = bf16_pair_bits(vsh[key * HD + col], vsh[(key + 1) * HD + col]);
    }
}

template <int HD>
__device__ static __forceinline__ void tc_load_b_pv_hd_f16(uint32_t b[4], const __half *vsh, int d0) {
    int lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < 4; r++) {
        int col = d0 + ((r >> 1) << 3) + (lane >> 2);
        int key = ((r & 1) << 3) + ((lane & 3) << 1);
        b[r] = f16_pair_bits(vsh[key * HD + col], vsh[(key + 1) * HD + col]);
    }
}

template <int HD>
__device__ static __forceinline__ void
bert_transpose_k_tile_bf16(__nv_bfloat16 *ktsh, const __nv_bfloat16 *krow, int tid, int nthreads) {
    enum { KT = 64 };
    for (int idx = tid; idx < KT * HD; idx += nthreads) {
        int kk = idx / HD;
        int d = idx - kk * HD;
        tc_store_swz_bf16<16>(ktsh + (kk >> 4) * 16 * HD, d, kk & 15, krow[idx]);
    }
}

template <int HD>
__device__ static __forceinline__ void
bert_transpose_k_tile_f16(__half *ktsh, const __half *krow, int tid, int nthreads) {
    enum { KT = 64 };
    for (int idx = tid; idx < KT * HD; idx += nthreads) {
        int kk = idx / HD;
        int d = idx - kk * HD;
        tc_store_swz_f16<16>(ktsh + (kk >> 4) * 16 * HD, d, kk & 15, krow[idx]);
    }
}

__device__ static __forceinline__ void tc_cp_async_cg_16(void *dst, const void *src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t smem = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem), "l"(src) : "memory");
#else
    (void)dst;
    (void)src;
#endif
}

__device__ static __forceinline__ void tc_cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" : : : "memory");
#endif
}

template <int N> __device__ static __forceinline__ void tc_cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
#endif
}

template <int HD, typename QKV_T, int DIRECT_KV>
__device__ static __forceinline__ int bert_stage_kv_tile_bf16(__nv_bfloat16 *ksh,
                                                              __nv_bfloat16 *vsh,
                                                              const QKV_T *qkv,
                                                              const __nv_bfloat16 *kexp,
                                                              const __nv_bfloat16 *vexp,
                                                              int start,
                                                              int L,
                                                              int k0,
                                                              int q_dim,
                                                              int qkv_dim,
                                                              int k_offset,
                                                              int v_offset,
                                                              size_t hv,
                                                              int kv_head,
                                                              int tid,
                                                              int nthreads) {
    enum { KT = 64, VEC = 8, SEGS = KT * HD / VEC };
    bool full_tile = k0 + KT <= L;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (full_tile) {
        for (int seg = tid; seg < SEGS; seg += nthreads) {
            int elem = seg * VEC;
            int kk = elem / HD;
            int d = elem - kk * HD;
            const __nv_bfloat16 *ksrc;
            const __nv_bfloat16 *vsrc;
            if constexpr (DIRECT_KV) {
                size_t base = (size_t)(start + k0 + kk) * qkv_dim + (size_t)kv_head * HD + d;
                ksrc = (const __nv_bfloat16 *)(qkv + base + k_offset);
                vsrc = (const __nv_bfloat16 *)(qkv + base + v_offset);
            } else {
                size_t base = (size_t)(start + k0 + kk) * q_dim + hv + d;
                ksrc = kexp + base;
                vsrc = vexp + base;
            }
            tc_cp_async_cg_16(ksh + kk * HD + d, ksrc);
            tc_cp_async_cg_16(vsh + kk * HD + d, vsrc);
        }
        tc_cp_async_commit_group();
        return 1;
    }
#endif
    for (int idx = tid; idx < KT * HD; idx += nthreads) {
        int kk = idx / HD;
        int d = idx - kk * HD;
        int key = k0 + kk;
        __nv_bfloat16 kval = __float2bfloat16(0.0f);
        __nv_bfloat16 vval = __float2bfloat16(0.0f);
        if (key < L) {
            if constexpr (DIRECT_KV) {
                size_t base = (size_t)(start + key) * qkv_dim + (size_t)kv_head * HD + d;
                kval = __float2bfloat16(load_act(qkv, base + k_offset));
                vval = __float2bfloat16(load_act(qkv, base + v_offset));
            } else {
                size_t base = (size_t)(start + key) * q_dim + hv + d;
                kval = kexp[base];
                vval = vexp[base];
            }
        }
        ksh[idx] = kval;
        vsh[idx] = vval;
    }
    return 0;
}

template <int HD, typename QKV_T, int DIRECT_KV>
__device__ static __forceinline__ int bert_stage_kv_tile_f16(__half *ksh,
                                                             __half *vsh,
                                                             const QKV_T *qkv,
                                                             const __half *kexp,
                                                             const __half *vexp,
                                                             int start,
                                                             int L,
                                                             int k0,
                                                             int q_dim,
                                                             int qkv_dim,
                                                             int k_offset,
                                                             int v_offset,
                                                             size_t hv,
                                                             int kv_head,
                                                             int tid,
                                                             int nthreads) {
    enum { KT = 64, VEC = 8, SEGS = KT * HD / VEC };
    bool full_tile = k0 + KT <= L;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (full_tile) {
        for (int seg = tid; seg < SEGS; seg += nthreads) {
            int elem = seg * VEC;
            int kk = elem / HD;
            int d = elem - kk * HD;
            const __half *ksrc;
            const __half *vsrc;
            if constexpr (DIRECT_KV) {
                size_t base = (size_t)(start + k0 + kk) * qkv_dim + (size_t)kv_head * HD + d;
                ksrc = (const __half *)(qkv + base + k_offset);
                vsrc = (const __half *)(qkv + base + v_offset);
            } else {
                size_t base = (size_t)(start + k0 + kk) * q_dim + hv + d;
                ksrc = kexp + base;
                vsrc = vexp + base;
            }
            tc_cp_async_cg_16(ksh + kk * HD + d, ksrc);
            tc_cp_async_cg_16(vsh + kk * HD + d, vsrc);
        }
        tc_cp_async_commit_group();
        return 1;
    }
#endif
    for (int idx = tid; idx < KT * HD; idx += nthreads) {
        int kk = idx / HD;
        int d = idx - kk * HD;
        int key = k0 + kk;
        __half kval = __float2half(0.0f);
        __half vval = __float2half(0.0f);
        if (key < L) {
            if constexpr (DIRECT_KV) {
                size_t base = (size_t)(start + key) * qkv_dim + (size_t)kv_head * HD + d;
                kval = __float2half(load_act(qkv, base + k_offset));
                vval = __float2half(load_act(qkv, base + v_offset));
            } else {
                size_t base = (size_t)(start + key) * q_dim + hv + d;
                kval = kexp[base];
                vval = vexp[base];
            }
        }
        ksh[idx] = kval;
        vsh[idx] = vval;
    }
    return 0;
}

// Qwen BF16 tensor-core flash attention. One block owns WARPS 16-query tiles
// that share a 16-key K/V staging tile. Q is stored with a grain-level XOR
// swizzle and K is staged transposed+swizzled so ldmatrix.x4.trans feeds the QK
// MMA fragments directly; P*V uses the validated BF16 pair loader.
template <int CAUSAL, int WARPS>
__global__ static void attn_stream_tc128_bf16_kv_ldm_kernel(__nv_bfloat16 *out,
                                                            const float *qkv,
                                                            const __nv_bfloat16 *kexp,
                                                            const __nv_bfloat16 *vexp,
                                                            const int *offsets,
                                                            int q_dim,
                                                            int qkv_dim,
                                                            int q_offset,
                                                            float scale) {
    enum { QT = 16, KT = 16, HD = 128, DCHUNKS = 8 };
    __shared__ __align__(16) __nv_bfloat16 qsh[WARPS * QT * HD];
    __shared__ __align__(16) __nv_bfloat16 ksh[KT * HD];
    __shared__ __align__(16) __nv_bfloat16 vsh[KT * HD];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.y;
    int tid = warp * 32 + lane;
    int nthreads = WARPS * 32;
    int block_q0 = blockIdx.x * (WARPS * QT);
    int head = blockIdx.y;
    int b = blockIdx.z;
    int start = offsets[b];
    int L = offsets[b + 1] - start;
    if (block_q0 >= L)
        return;
    int q0 = block_q0 + warp * QT;
    size_t hv = (size_t)head * HD;
    __nv_bfloat16 *myq = qsh + warp * QT * HD;

    for (int idx = lane; idx < QT * HD; idx += 32) {
        int qr = idx / HD;
        int d = idx - qr * HD;
        int row = q0 + qr;
        float qv = row < L ? qkv[(size_t)(start + row) * qkv_dim + q_offset + hv + d] : 0.0f;
        tc_store_swz_bf16<128>(myq, qr, d, __float2bfloat16(qv));
    }

    float o[DCHUNKS][8];
#pragma unroll
    for (int dc = 0; dc < DCHUNKS; dc++)
#pragma unroll
        for (int r = 0; r < 8; r++)
            o[dc][r] = 0.0f;
    float m[2] = {-3.402823466e+38F, -3.402823466e+38F};
    float denom[2] = {0.0f, 0.0f};

    int key_hi = L;
    if (CAUSAL) {
        int lim = block_q0 + WARPS * QT;
        if (lim < key_hi)
            key_hi = lim;
    }

    for (int k0 = 0; k0 < key_hi; k0 += KT) {
        for (int idx = tid; idx < KT * HD; idx += nthreads) {
            int kk = idx / HD;
            int d = idx - kk * HD;
            int key = k0 + kk;
            size_t kb = (size_t)(start + key) * q_dim + hv + d;
            __nv_bfloat16 kval = key < L ? kexp[kb] : __float2bfloat16(0.0f);
            __nv_bfloat16 vval = key < L ? vexp[kb] : __float2bfloat16(0.0f);
            tc_store_swz_bf16<16>(ksh, d, kk, kval);
            vsh[idx] = vval;
        }
        __syncthreads();

        float s[8];
#pragma unroll
        for (int r = 0; r < 8; r++)
            s[r] = 0.0f;
#pragma unroll
        for (int d0 = 0; d0 < HD; d0 += 16) {
            uint32_t a[4], bb[4];
            tc_load_a_q_ldmatrix_swz_bf16(a, myq, d0);
            tc_load_b_qk_ldmatrix_swz_bf16(bb, ksh, d0);
            mma_m16n16k16_bf16(s, a, bb);
        }

        const float neg_inf = -3.402823466e+38F;
#pragma unroll
        for (int r = 0; r < 8; r++) {
            int qr = (lane >> 2) + (((r % 4) >> 1) << 3);
            int kc = ((lane & 3) << 1) + ((r >> 2) << 3) + (r & 1);
            int qrow = q0 + qr;
            int key = k0 + kc;
            if (qrow < L && key < L && (!CAUSAL || key <= qrow))
                s[r] *= scale;
            else
                s[r] = neg_inf;
        }

#pragma unroll
        for (int j = 0; j < 2; j++) {
            int r0 = j * 2;
            float tile_m = fmaxf(fmaxf(s[r0 + 0], s[r0 + 1]), fmaxf(s[r0 + 4], s[r0 + 5]));
            float m_new = fmaxf(m[j], warp_group4_max(tile_m));
            float corr = (m_new == neg_inf) ? 1.0f : __expf(m[j] - m_new);
            denom[j] *= corr;
#pragma unroll
            for (int dc = 0; dc < DCHUNKS; dc++) {
                o[dc][r0 + 0] *= corr;
                o[dc][r0 + 1] *= corr;
                o[dc][r0 + 4] *= corr;
                o[dc][r0 + 5] *= corr;
            }
            float p0 = (s[r0 + 0] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 0] - m_new);
            float p1 = (s[r0 + 1] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 1] - m_new);
            float p4 = (s[r0 + 4] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 4] - m_new);
            float p5 = (s[r0 + 5] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 5] - m_new);
            s[r0 + 0] = p0;
            s[r0 + 1] = p1;
            s[r0 + 4] = p4;
            s[r0 + 5] = p5;
            denom[j] += warp_group4_sum(p0 + p1 + p4 + p5);
            m[j] = m_new;
        }

        uint32_t pfrag[4];
        tc_probs_to_a_bf16(pfrag, s);
#pragma unroll
        for (int dc = 0; dc < DCHUNKS; dc++) {
            uint32_t vb[4];
            tc_load_b_pv_bf16(vb, vsh, dc * 16);
            mma_m16n16k16_bf16(o[dc], pfrag, vb);
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < 2; j++) {
        int qrow = q0 + (lane >> 2) + j * 8;
        if (qrow >= L)
            continue;
        float inv = denom[j] > 0.0f ? 1.0f / denom[j] : 0.0f;
        size_t ob = (size_t)(start + qrow) * q_dim + hv;
#pragma unroll
        for (int dc = 0; dc < DCHUNKS; dc++) {
            int col = dc * 16 + ((lane & 3) << 1);
            store_act(out, ob + col + 0, o[dc][j * 2 + 0] * inv);
            store_act(out, ob + col + 1, o[dc][j * 2 + 1] * inv);
            store_act(out, ob + col + 8, o[dc][j * 2 + 4] * inv);
            store_act(out, ob + col + 9, o[dc][j * 2 + 5] * inv);
        }
    }
}

// BF16 tensor-core flash attention for the BERT shape: head_dim 32/64, BIDIRECTIONAL
// (no causal mask). The tc3 kernel for the Qwen shape (head_dim 128, K/V-sharing,
// ldmatrix QK, online softmax) generalized to smaller BERT head sizes: fewer
// d-chunks, an HD-wide Q swizzle and V row stride, and an every-key loop with
// only the length mask. Replaces the materialized N×N scores/probs path for BERT: the
// scores live in registers and never reach global memory, removing the O(n^2)
// HBM that dominates BERT attention at batch 32. One block = WARPS query tiles
// sharing one K/V staging tile; the math is otherwise identical to tc3.
template <int HD, int WARPS, typename QKV_T, int DIRECT_KV, int PIPELINED>
__global__ static void attn_stream_bert_flash_kernel(__nv_bfloat16 *out,
                                                     const QKV_T *qkv,
                                                     const __nv_bfloat16 *kexp,
                                                     const __nv_bfloat16 *vexp,
                                                     const int *offsets,
                                                     int q_dim,
                                                     int qkv_dim,
                                                     int q_offset,
                                                     int k_offset,
                                                     int v_offset,
                                                     int n_heads,
                                                     int n_kv_heads,
                                                     float scale) {
    enum { QT = 16, KT = 64, DCHUNKS = HD / 16, KSUB = KT / 16 };
    enum { Q_ELEMS = WARPS * QT * HD, KV_LEGACY_ELEMS = Q_ELEMS + 2 * KT * HD };
    enum { KV_PIPE_ELEMS = 5 * KT * HD };
    enum {
        SH_ELEMS = PIPELINED ? ((Q_ELEMS > KV_PIPE_ELEMS) ? Q_ELEMS : KV_PIPE_ELEMS) : KV_LEGACY_ELEMS
    };
    __shared__ __align__(16) __nv_bfloat16 shmem[SH_ELEMS];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.y;
    int tid = warp * 32 + lane;
    int nthreads = WARPS * 32;
    int block_q0 = blockIdx.x * (WARPS * QT);
    int head = blockIdx.y;
    int b = blockIdx.z;
    int start = offsets[b];
    int L = offsets[b + 1] - start;
    if (block_q0 >= L)
        return;
    int q0 = block_q0 + warp * QT;
    size_t hv = (size_t)head * HD;
    int kv_head = head / (n_heads / n_kv_heads);
    __nv_bfloat16 *qsh = shmem;
    __nv_bfloat16 *myq = qsh + warp * QT * HD;

    for (int idx = lane; idx < QT * HD; idx += 32) {
        int qr = idx / HD;
        int d = idx - qr * HD;
        int row = q0 + qr;
        float qv =
            row < L ? load_act(qkv, (size_t)(start + row) * qkv_dim + q_offset + hv + d) : 0.0f;
        tc_store_swz_bf16<HD>(myq, qr, d, __float2bfloat16(qv));
    }
    // Each warp fills its own Q tile cooperatively across lanes, then reads it
    // back with ldmatrix. ldmatrix pulls shared locations written by other lanes
    // of the warp, so the cooperative stores must be visible first; without this
    // warp barrier the load can observe stale Q on independent-thread-scheduling
    // GPUs (a nondeterministic shared-memory hazard).
    __syncwarp();

    // Q fragments do not change across keys: load them once and reuse them for
    // every K sub-tile, instead of re-issuing ldmatrix on every K step.
    uint32_t a_q[DCHUNKS][4];
#pragma unroll
    for (int dc = 0; dc < DCHUNKS; dc++)
        tc_load_a_q_ldmatrix_swz_hd<HD>(a_q[dc], myq, dc * 16);
    if constexpr (PIPELINED)
        __syncthreads();

    float o[DCHUNKS][8];
#pragma unroll
    for (int dc = 0; dc < DCHUNKS; dc++)
#pragma unroll
        for (int r = 0; r < 8; r++)
            o[dc][r] = 0.0f;
    float m[2] = {-3.402823466e+38F, -3.402823466e+38F};
    float denom[2] = {0.0f, 0.0f};

    if constexpr (PIPELINED) {
        __nv_bfloat16 *kstage0 = shmem;
        __nv_bfloat16 *kstage1 = kstage0 + KT * HD;
        __nv_bfloat16 *vstage0 = kstage1 + KT * HD;
        __nv_bfloat16 *vstage1 = vstage0 + KT * HD;
        __nv_bfloat16 *ktsh = vstage1 + KT * HD;

        int stage = 0;
        int async_stage = bert_stage_kv_tile_bf16<HD, QKV_T, DIRECT_KV>(
            kstage0, vstage0, qkv, kexp, vexp, start, L, 0, q_dim, qkv_dim, k_offset, v_offset, hv,
            kv_head, tid, nthreads);

        // K/V staging is double-buffered. While one 64-key tile feeds the
        // unchanged QK/softmax/PV recurrence, the next tile is already moving
        // from global to the other shared stage.
        for (int k0 = 0; k0 < L; k0 += KT) { // bidirectional: every query attends every key
            if (async_stage)
                tc_cp_async_wait_group<0>();
            __syncthreads();

            __nv_bfloat16 *krow = stage ? kstage1 : kstage0;
            __nv_bfloat16 *vsh = stage ? vstage1 : vstage0;
            bert_transpose_k_tile_bf16<HD>(ktsh, krow, tid, nthreads);
            __syncthreads();

            int next_k0 = k0 + KT;
            int next_stage = stage ^ 1;
            int async_next = 0;
            if (next_k0 < L) {
                async_next = bert_stage_kv_tile_bf16<HD, QKV_T, DIRECT_KV>(
                    next_stage ? kstage1 : kstage0, next_stage ? vstage1 : vstage0, qkv, kexp, vexp,
                    start, L, next_k0, q_dim, qkv_dim, k_offset, v_offset, hv, kv_head, tid,
                    nthreads);
            }

#pragma unroll
            for (int sub = 0; sub < KSUB; sub++) {
                const __nv_bfloat16 *ksh_s = ktsh + sub * 16 * HD;
                const __nv_bfloat16 *vsh_s = vsh + sub * 16 * HD;
                int ksub0 = k0 + sub * 16;
                if (ksub0 >= L)
                    break;

                float s[8];
#pragma unroll
                for (int r = 0; r < 8; r++)
                    s[r] = 0.0f;
#pragma unroll
                for (int d0 = 0; d0 < HD; d0 += 16) {
                    uint32_t bb[4];
                    tc_load_b_qk_ldmatrix_swz_bf16(bb, ksh_s, d0);
                    mma_m16n16k16_bf16(s, a_q[d0 >> 4], bb);
                }

                const float neg_inf = -3.402823466e+38F;
#pragma unroll
                for (int r = 0; r < 8; r++) {
                    int qr = (lane >> 2) + (((r % 4) >> 1) << 3);
                    int kc = ((lane & 3) << 1) + ((r >> 2) << 3) + (r & 1);
                    int qrow = q0 + qr;
                    int key = ksub0 + kc;
                    if (qrow < L && key < L)
                        s[r] *= scale;
                    else
                        s[r] = neg_inf;
                }

#pragma unroll
                for (int j = 0; j < 2; j++) {
                    int r0 = j * 2;
                    float tile_m = fmaxf(fmaxf(s[r0 + 0], s[r0 + 1]), fmaxf(s[r0 + 4], s[r0 + 5]));
                    float m_new = fmaxf(m[j], warp_group4_max(tile_m));
                    float corr = (m_new == neg_inf) ? 1.0f : __expf(m[j] - m_new);
                    denom[j] *= corr;
#pragma unroll
                    for (int dc = 0; dc < DCHUNKS; dc++) {
                        o[dc][r0 + 0] *= corr;
                        o[dc][r0 + 1] *= corr;
                        o[dc][r0 + 4] *= corr;
                        o[dc][r0 + 5] *= corr;
                    }
                    float p0 =
                        (s[r0 + 0] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 0] - m_new);
                    float p1 =
                        (s[r0 + 1] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 1] - m_new);
                    float p4 =
                        (s[r0 + 4] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 4] - m_new);
                    float p5 =
                        (s[r0 + 5] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 5] - m_new);
                    s[r0 + 0] = p0;
                    s[r0 + 1] = p1;
                    s[r0 + 4] = p4;
                    s[r0 + 5] = p5;
                    denom[j] += warp_group4_sum(p0 + p1 + p4 + p5);
                    m[j] = m_new;
                }

                uint32_t pfrag[4];
                tc_probs_to_a_bf16(pfrag, s);
#pragma unroll
                for (int dc = 0; dc < DCHUNKS; dc++) {
                    uint32_t vb[4];
                    tc_load_b_pv_hd<HD>(vb, vsh_s, dc * 16);
                    mma_m16n16k16_bf16(o[dc], pfrag, vb);
                }
            }
            stage = next_stage;
            async_stage = async_next;
        }
    } else {
        __nv_bfloat16 *ksh = shmem + Q_ELEMS;
        __nv_bfloat16 *vsh = ksh + KT * HD;

        for (int k0 = 0; k0 < L; k0 += KT) {
            for (int idx = tid; idx < KT * HD; idx += nthreads) {
                int kk = idx / HD;
                int d = idx - kk * HD;
                int key = k0 + kk;
                __nv_bfloat16 kval = __float2bfloat16(0.0f);
                __nv_bfloat16 vval = __float2bfloat16(0.0f);
                if (key < L) {
                    if (DIRECT_KV) {
                        size_t base = (size_t)(start + key) * qkv_dim + (size_t)kv_head * HD + d;
                        kval = __float2bfloat16(load_act(qkv, base + k_offset));
                        vval = __float2bfloat16(load_act(qkv, base + v_offset));
                    } else {
                        size_t kb = (size_t)(start + key) * q_dim + hv + d;
                        kval = kexp[kb];
                        vval = vexp[kb];
                    }
                }
                tc_store_swz_bf16<16>(ksh + (kk >> 4) * 16 * HD, d, kk & 15, kval);
                vsh[idx] = vval;
            }
            __syncthreads();

#pragma unroll
            for (int sub = 0; sub < KSUB; sub++) {
                const __nv_bfloat16 *ksh_s = ksh + sub * 16 * HD;
                const __nv_bfloat16 *vsh_s = vsh + sub * 16 * HD;
                int ksub0 = k0 + sub * 16;
                if (ksub0 >= L)
                    break;

                float s[8];
#pragma unroll
                for (int r = 0; r < 8; r++)
                    s[r] = 0.0f;
#pragma unroll
                for (int d0 = 0; d0 < HD; d0 += 16) {
                    uint32_t bb[4];
                    tc_load_b_qk_ldmatrix_swz_bf16(bb, ksh_s, d0);
                    mma_m16n16k16_bf16(s, a_q[d0 >> 4], bb);
                }

                const float neg_inf = -3.402823466e+38F;
#pragma unroll
                for (int r = 0; r < 8; r++) {
                    int qr = (lane >> 2) + (((r % 4) >> 1) << 3);
                    int kc = ((lane & 3) << 1) + ((r >> 2) << 3) + (r & 1);
                    int qrow = q0 + qr;
                    int key = ksub0 + kc;
                    if (qrow < L && key < L)
                        s[r] *= scale;
                    else
                        s[r] = neg_inf;
                }

#pragma unroll
                for (int j = 0; j < 2; j++) {
                    int r0 = j * 2;
                    float tile_m = fmaxf(fmaxf(s[r0 + 0], s[r0 + 1]), fmaxf(s[r0 + 4], s[r0 + 5]));
                    float m_new = fmaxf(m[j], warp_group4_max(tile_m));
                    float corr = (m_new == neg_inf) ? 1.0f : __expf(m[j] - m_new);
                    denom[j] *= corr;
#pragma unroll
                    for (int dc = 0; dc < DCHUNKS; dc++) {
                        o[dc][r0 + 0] *= corr;
                        o[dc][r0 + 1] *= corr;
                        o[dc][r0 + 4] *= corr;
                        o[dc][r0 + 5] *= corr;
                    }
                    float p0 =
                        (s[r0 + 0] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 0] - m_new);
                    float p1 =
                        (s[r0 + 1] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 1] - m_new);
                    float p4 =
                        (s[r0 + 4] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 4] - m_new);
                    float p5 =
                        (s[r0 + 5] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 5] - m_new);
                    s[r0 + 0] = p0;
                    s[r0 + 1] = p1;
                    s[r0 + 4] = p4;
                    s[r0 + 5] = p5;
                    denom[j] += warp_group4_sum(p0 + p1 + p4 + p5);
                    m[j] = m_new;
                }

                uint32_t pfrag[4];
                tc_probs_to_a_bf16(pfrag, s);
#pragma unroll
                for (int dc = 0; dc < DCHUNKS; dc++) {
                    uint32_t vb[4];
                    tc_load_b_pv_hd<HD>(vb, vsh_s, dc * 16);
                    mma_m16n16k16_bf16(o[dc], pfrag, vb);
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < 2; j++) {
        int qrow = q0 + (lane >> 2) + j * 8;
        if (qrow >= L)
            continue;
        float inv = denom[j] > 0.0f ? 1.0f / denom[j] : 0.0f;
        size_t ob = (size_t)(start + qrow) * q_dim + hv;
#pragma unroll
        for (int dc = 0; dc < DCHUNKS; dc++) {
            int col = dc * 16 + ((lane & 3) << 1);
            store_act(out, ob + col + 0, o[dc][j * 2 + 0] * inv);
            store_act(out, ob + col + 1, o[dc][j * 2 + 1] * inv);
            store_act(out, ob + col + 8, o[dc][j * 2 + 4] * inv);
            store_act(out, ob + col + 9, o[dc][j * 2 + 5] * inv);
        }
    }
}

// FP16 sibling of the BERT flash kernel. The tiling, online softmax, and
// register layout intentionally match the BF16 kernel above; only the staged
// operand type and MMA instruction change.
template <int HD, int WARPS, typename QKV_T, int DIRECT_KV, int PIPELINED>
__global__ static void attn_stream_bert_flash_f16_kernel(__half *out,
                                                         const QKV_T *qkv,
                                                         const __half *kexp,
                                                         const __half *vexp,
                                                         const int *offsets,
                                                         int q_dim,
                                                         int qkv_dim,
                                                         int q_offset,
                                                         int k_offset,
                                                         int v_offset,
                                                         int n_heads,
                                                         int n_kv_heads,
                                                         float scale) {
    enum { QT = 16, KT = 64, DCHUNKS = HD / 16, KSUB = KT / 16 };
    enum { Q_ELEMS = WARPS * QT * HD, KV_LEGACY_ELEMS = Q_ELEMS + 2 * KT * HD };
    enum { KV_PIPE_ELEMS = 5 * KT * HD };
    enum {
        SH_ELEMS = PIPELINED ? ((Q_ELEMS > KV_PIPE_ELEMS) ? Q_ELEMS : KV_PIPE_ELEMS) : KV_LEGACY_ELEMS
    };
    __shared__ __align__(16) __half shmem[SH_ELEMS];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.y;
    int tid = warp * 32 + lane;
    int nthreads = WARPS * 32;
    int block_q0 = blockIdx.x * (WARPS * QT);
    int head = blockIdx.y;
    int b = blockIdx.z;
    int start = offsets[b];
    int L = offsets[b + 1] - start;
    if (block_q0 >= L)
        return;
    int q0 = block_q0 + warp * QT;
    size_t hv = (size_t)head * HD;
    int kv_head = head / (n_heads / n_kv_heads);
    __half *qsh = shmem;
    __half *myq = qsh + warp * QT * HD;

    for (int idx = lane; idx < QT * HD; idx += 32) {
        int qr = idx / HD;
        int d = idx - qr * HD;
        int row = q0 + qr;
        float qv =
            row < L ? load_act(qkv, (size_t)(start + row) * qkv_dim + q_offset + hv + d) : 0.0f;
        tc_store_swz_f16<HD>(myq, qr, d, __float2half(qv));
    }
    // See the bf16 sibling: ldmatrix reads Q written by other lanes of the warp,
    // so the cooperative stores need a warp barrier to be visible first.
    __syncwarp();

    // Q fragments do not change across keys: load them once and reuse them for
    // every K sub-tile, instead of re-issuing ldmatrix on every K step.
    uint32_t a_q[DCHUNKS][4];
#pragma unroll
    for (int dc = 0; dc < DCHUNKS; dc++)
        tc_load_a_q_ldmatrix_swz_hd_f16<HD>(a_q[dc], myq, dc * 16);
    if constexpr (PIPELINED)
        __syncthreads();

    float o[DCHUNKS][8];
#pragma unroll
    for (int dc = 0; dc < DCHUNKS; dc++)
#pragma unroll
        for (int r = 0; r < 8; r++)
            o[dc][r] = 0.0f;
    float m[2] = {-3.402823466e+38F, -3.402823466e+38F};
    float denom[2] = {0.0f, 0.0f};

    if constexpr (PIPELINED) {
        __half *kstage0 = shmem;
        __half *kstage1 = kstage0 + KT * HD;
        __half *vstage0 = kstage1 + KT * HD;
        __half *vstage1 = vstage0 + KT * HD;
        __half *ktsh = vstage1 + KT * HD;

        int stage = 0;
        int async_stage = bert_stage_kv_tile_f16<HD, QKV_T, DIRECT_KV>(
            kstage0, vstage0, qkv, kexp, vexp, start, L, 0, q_dim, qkv_dim, k_offset, v_offset, hv,
            kv_head, tid, nthreads);

        // K/V staging is double-buffered. While one 64-key tile feeds the
        // unchanged QK/softmax/PV recurrence, the next tile is already moving
        // from global to the other shared stage.
        for (int k0 = 0; k0 < L; k0 += KT) {
            if (async_stage)
                tc_cp_async_wait_group<0>();
            __syncthreads();

            __half *krow = stage ? kstage1 : kstage0;
            __half *vsh = stage ? vstage1 : vstage0;
            bert_transpose_k_tile_f16<HD>(ktsh, krow, tid, nthreads);
            __syncthreads();

            int next_k0 = k0 + KT;
            int next_stage = stage ^ 1;
            int async_next = 0;
            if (next_k0 < L) {
                async_next = bert_stage_kv_tile_f16<HD, QKV_T, DIRECT_KV>(
                    next_stage ? kstage1 : kstage0, next_stage ? vstage1 : vstage0, qkv, kexp, vexp,
                    start, L, next_k0, q_dim, qkv_dim, k_offset, v_offset, hv, kv_head, tid,
                    nthreads);
            }

#pragma unroll
            for (int sub = 0; sub < KSUB; sub++) {
                const __half *ksh_s = ktsh + sub * 16 * HD;
                const __half *vsh_s = vsh + sub * 16 * HD;
                int ksub0 = k0 + sub * 16;
                if (ksub0 >= L)
                    break;

                float s[8];
#pragma unroll
                for (int r = 0; r < 8; r++)
                    s[r] = 0.0f;
#pragma unroll
                for (int d0 = 0; d0 < HD; d0 += 16) {
                    uint32_t bb[4];
                    tc_load_b_qk_ldmatrix_swz_f16(bb, ksh_s, d0);
                    mma_m16n16k16_f16(s, a_q[d0 >> 4], bb);
                }

                const float neg_inf = -3.402823466e+38F;
#pragma unroll
                for (int r = 0; r < 8; r++) {
                    int qr = (lane >> 2) + (((r % 4) >> 1) << 3);
                    int kc = ((lane & 3) << 1) + ((r >> 2) << 3) + (r & 1);
                    int qrow = q0 + qr;
                    int key = ksub0 + kc;
                    if (qrow < L && key < L)
                        s[r] *= scale;
                    else
                        s[r] = neg_inf;
                }

#pragma unroll
                for (int j = 0; j < 2; j++) {
                    int r0 = j * 2;
                    float tile_m = fmaxf(fmaxf(s[r0 + 0], s[r0 + 1]), fmaxf(s[r0 + 4], s[r0 + 5]));
                    float m_new = fmaxf(m[j], warp_group4_max(tile_m));
                    float corr = (m_new == neg_inf) ? 1.0f : __expf(m[j] - m_new);
                    denom[j] *= corr;
#pragma unroll
                    for (int dc = 0; dc < DCHUNKS; dc++) {
                        o[dc][r0 + 0] *= corr;
                        o[dc][r0 + 1] *= corr;
                        o[dc][r0 + 4] *= corr;
                        o[dc][r0 + 5] *= corr;
                    }
                    float p0 =
                        (s[r0 + 0] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 0] - m_new);
                    float p1 =
                        (s[r0 + 1] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 1] - m_new);
                    float p4 =
                        (s[r0 + 4] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 4] - m_new);
                    float p5 =
                        (s[r0 + 5] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 5] - m_new);
                    s[r0 + 0] = p0;
                    s[r0 + 1] = p1;
                    s[r0 + 4] = p4;
                    s[r0 + 5] = p5;
                    denom[j] += warp_group4_sum(p0 + p1 + p4 + p5);
                    m[j] = m_new;
                }

                uint32_t pfrag[4];
                tc_probs_to_a_f16(pfrag, s);
#pragma unroll
                for (int dc = 0; dc < DCHUNKS; dc++) {
                    uint32_t vb[4];
                    tc_load_b_pv_hd_f16<HD>(vb, vsh_s, dc * 16);
                    mma_m16n16k16_f16(o[dc], pfrag, vb);
                }
            }
            stage = next_stage;
            async_stage = async_next;
        }
    } else {
        __half *ksh = shmem + Q_ELEMS;
        __half *vsh = ksh + KT * HD;

        for (int k0 = 0; k0 < L; k0 += KT) {
            for (int idx = tid; idx < KT * HD; idx += nthreads) {
                int kk = idx / HD;
                int d = idx - kk * HD;
                int key = k0 + kk;
                __half kval = __float2half(0.0f);
                __half vval = __float2half(0.0f);
                if (key < L) {
                    if (DIRECT_KV) {
                        size_t base = (size_t)(start + key) * qkv_dim + (size_t)kv_head * HD + d;
                        kval = __float2half(load_act(qkv, base + k_offset));
                        vval = __float2half(load_act(qkv, base + v_offset));
                    } else {
                        size_t kb = (size_t)(start + key) * q_dim + hv + d;
                        kval = kexp[kb];
                        vval = vexp[kb];
                    }
                }
                tc_store_swz_f16<16>(ksh + (kk >> 4) * 16 * HD, d, kk & 15, kval);
                vsh[idx] = vval;
            }
            __syncthreads();

#pragma unroll
            for (int sub = 0; sub < KSUB; sub++) {
                const __half *ksh_s = ksh + sub * 16 * HD;
                const __half *vsh_s = vsh + sub * 16 * HD;
                int ksub0 = k0 + sub * 16;
                if (ksub0 >= L)
                    break;

                float s[8];
#pragma unroll
                for (int r = 0; r < 8; r++)
                    s[r] = 0.0f;
#pragma unroll
                for (int d0 = 0; d0 < HD; d0 += 16) {
                    uint32_t bb[4];
                    tc_load_b_qk_ldmatrix_swz_f16(bb, ksh_s, d0);
                    mma_m16n16k16_f16(s, a_q[d0 >> 4], bb);
                }

                const float neg_inf = -3.402823466e+38F;
#pragma unroll
                for (int r = 0; r < 8; r++) {
                    int qr = (lane >> 2) + (((r % 4) >> 1) << 3);
                    int kc = ((lane & 3) << 1) + ((r >> 2) << 3) + (r & 1);
                    int qrow = q0 + qr;
                    int key = ksub0 + kc;
                    if (qrow < L && key < L)
                        s[r] *= scale;
                    else
                        s[r] = neg_inf;
                }

#pragma unroll
                for (int j = 0; j < 2; j++) {
                    int r0 = j * 2;
                    float tile_m = fmaxf(fmaxf(s[r0 + 0], s[r0 + 1]), fmaxf(s[r0 + 4], s[r0 + 5]));
                    float m_new = fmaxf(m[j], warp_group4_max(tile_m));
                    float corr = (m_new == neg_inf) ? 1.0f : __expf(m[j] - m_new);
                    denom[j] *= corr;
#pragma unroll
                    for (int dc = 0; dc < DCHUNKS; dc++) {
                        o[dc][r0 + 0] *= corr;
                        o[dc][r0 + 1] *= corr;
                        o[dc][r0 + 4] *= corr;
                        o[dc][r0 + 5] *= corr;
                    }
                    float p0 =
                        (s[r0 + 0] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 0] - m_new);
                    float p1 =
                        (s[r0 + 1] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 1] - m_new);
                    float p4 =
                        (s[r0 + 4] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 4] - m_new);
                    float p5 =
                        (s[r0 + 5] == neg_inf || m_new == neg_inf) ? 0.0f : __expf(s[r0 + 5] - m_new);
                    s[r0 + 0] = p0;
                    s[r0 + 1] = p1;
                    s[r0 + 4] = p4;
                    s[r0 + 5] = p5;
                    denom[j] += warp_group4_sum(p0 + p1 + p4 + p5);
                    m[j] = m_new;
                }

                uint32_t pfrag[4];
                tc_probs_to_a_f16(pfrag, s);
#pragma unroll
                for (int dc = 0; dc < DCHUNKS; dc++) {
                    uint32_t vb[4];
                    tc_load_b_pv_hd_f16<HD>(vb, vsh_s, dc * 16);
                    mma_m16n16k16_f16(o[dc], pfrag, vb);
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < 2; j++) {
        int qrow = q0 + (lane >> 2) + j * 8;
        if (qrow >= L)
            continue;
        float inv = denom[j] > 0.0f ? 1.0f / denom[j] : 0.0f;
        size_t ob = (size_t)(start + qrow) * q_dim + hv;
#pragma unroll
        for (int dc = 0; dc < DCHUNKS; dc++) {
            int col = dc * 16 + ((lane & 3) << 1);
            store_act(out, ob + col + 0, o[dc][j * 2 + 0] * inv);
            store_act(out, ob + col + 1, o[dc][j * 2 + 1] * inv);
            store_act(out, ob + col + 8, o[dc][j * 2 + 4] * inv);
            store_act(out, ob + col + 9, o[dc][j * 2 + 5] * inv);
        }
    }
}

__global__ static void
mean_pool_kernel(float *out, const float *x, const int *offsets, int batch, int hidden) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int start = offsets[b];
    int end = offsets[b + 1];
    int len = end - start;
    for (int d = tid; d < hidden; d += blockDim.x) {
        float sum = 0.0f;
        for (int t = start; t < end; t++)
            sum += x[(size_t)t * hidden + d];
        out[(size_t)b * hidden + d] = sum / (float)len;
    }
}

__global__ static void
last_pool_kernel(float *out, const float *x, const int *offsets, int batch, int hidden) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int last = offsets[b + 1] - 1;
    for (int d = tid; d < hidden; d += blockDim.x)
        out[(size_t)b * hidden + d] = x[(size_t)last * hidden + d];
}

/* CLS pooling: gather the first token of each packed sequence. */
__global__ static void
first_pool_kernel(float *out, const float *x, const int *offsets, int batch, int hidden) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int first = offsets[b];
    for (int d = tid; d < hidden; d += blockDim.x)
        out[(size_t)b * hidden + d] = x[(size_t)first * hidden + d];
}

__global__ static void span_pool_kernel(
    float *out, const float *x, const int *starts, const int *lens, int n_spans, int hidden) {
    int s = blockIdx.x;
    int tid = threadIdx.x;
    int start = starts[s];
    int len = lens[s];
    for (int d = tid; d < hidden; d += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t < len; t++)
            sum += x[(size_t)(start + t) * hidden + d];
        out[(size_t)s * hidden + d] = sum / (float)len;
    }
}

// BERT embedding LayerNorm: token lookup + absolute position + token type +
// LayerNorm in one row kernel. It avoids writing the pre-normalized embedding
// row to global memory only to read it back for LayerNorm.
__global__ static void bert_embed_layer_norm_kernel(float *out,
                                                    __nv_bfloat16 *out_bf16,
                                                    const int *ids,
                                                    const int *positions,
                                                    const void *emb,
                                                    int emb_bf16,
                                                    const float *pos_emb,
                                                    const float *token_type,
                                                    const float *gamma,
                                                    const float *beta,
                                                    int total,
                                                    int hidden,
                                                    int vocab,
                                                    float eps) {
    extern __shared__ float rowbuf[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= total)
        return;
    int id = ids[row];
    int pos = positions[row];

    float s = 0.0f;
    for (int d = tid; d < hidden; d += blockDim.x) {
        float tok = 0.0f;
        if (id >= 0 && id < vocab)
            tok = load_embedding_value(emb, emb_bf16, (size_t)id * hidden + d);
        float v = tok + (pos_emb[(size_t)pos * hidden + d] + token_type[d]);
        rowbuf[d] = v;
        s += v;
    }
    float mean = block_sum(s) / (float)hidden;
    float vs = 0.0f;
    for (int d = tid; d < hidden; d += blockDim.x) {
        float c = rowbuf[d] - mean;
        vs += c * c;
    }
    float inv = rsqrtf(block_sum(vs) / (float)hidden + eps);
    float *o = out + (size_t)row * hidden;
    __nv_bfloat16 *ob = out_bf16 ? out_bf16 + (size_t)row * hidden : nullptr;
    for (int d = tid; d < hidden; d += blockDim.x) {
        float nv = (rowbuf[d] - mean) * inv * gamma[d] + beta[d];
        o[d] = nv;
        if (ob)
            ob[d] = __float2bfloat16(nv);
    }
}

__global__ static void bert_embed_layer_norm_bf16_kernel(__nv_bfloat16 *out,
                                                         const int *ids,
                                                         const int *positions,
                                                         const void *emb,
                                                         int emb_bf16,
                                                         const float *pos_emb,
                                                         const float *token_type,
                                                         const float *gamma,
                                                         const float *beta,
                                                         int total,
                                                         int hidden,
                                                         int vocab,
                                                         float eps) {
    extern __shared__ float rowbuf[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= total)
        return;
    int id = ids[row];
    int pos = positions[row];

    float s = 0.0f;
    for (int d = tid; d < hidden; d += blockDim.x) {
        float tok = 0.0f;
        if (id >= 0 && id < vocab)
            tok = load_embedding_value(emb, emb_bf16, (size_t)id * hidden + d);
        float v = tok + (pos_emb[(size_t)pos * hidden + d] + token_type[d]);
        rowbuf[d] = v;
        s += v;
    }
    float mean = block_sum(s) / (float)hidden;
    float vs = 0.0f;
    for (int d = tid; d < hidden; d += blockDim.x) {
        float c = rowbuf[d] - mean;
        vs += c * c;
    }
    float inv = rsqrtf(block_sum(vs) / (float)hidden + eps);
    __nv_bfloat16 *o = out + (size_t)row * hidden;
    for (int d = tid; d < hidden; d += blockDim.x) {
        float nv = (rowbuf[d] - mean) * inv * gamma[d] + beta[d];
        o[d] = __float2bfloat16(nv);
    }
}

__global__ static void bert_embed_layer_norm_f16_act_kernel(float *out,
                                                            __half *out_f16,
                                                            const int *ids,
                                                            const int *positions,
                                                            const void *emb,
                                                            int emb_dtype,
                                                            const float *pos_emb,
                                                            const float *token_type,
                                                            const float *gamma,
                                                            const float *beta,
                                                            int total,
                                                            int hidden,
                                                            int vocab,
                                                            float eps) {
    extern __shared__ float rowbuf[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= total)
        return;
    int id = ids[row];
    int pos = positions[row];

    float s = 0.0f;
    for (int d = tid; d < hidden; d += blockDim.x) {
        float tok = 0.0f;
        if (id >= 0 && id < vocab)
            tok = load_embedding_value(emb, emb_dtype, (size_t)id * hidden + d);
        float v = tok + (pos_emb[(size_t)pos * hidden + d] + token_type[d]);
        rowbuf[d] = v;
        s += v;
    }
    float mean = block_sum(s) / (float)hidden;
    float vs = 0.0f;
    for (int d = tid; d < hidden; d += blockDim.x) {
        float c = rowbuf[d] - mean;
        vs += c * c;
    }
    float inv = rsqrtf(block_sum(vs) / (float)hidden + eps);
    float *o = out + (size_t)row * hidden;
    __half *oh = out_f16 ? out_f16 + (size_t)row * hidden : nullptr;
    for (int d = tid; d < hidden; d += blockDim.x) {
        float nv = (rowbuf[d] - mean) * inv * gamma[d] + beta[d];
        o[d] = nv;
        if (oh)
            oh[d] = __float2half(nv);
    }
}

__global__ static void bert_embed_layer_norm_f16_kernel(__half *out,
                                                        const int *ids,
                                                        const int *positions,
                                                        const void *emb,
                                                        int emb_dtype,
                                                        const float *pos_emb,
                                                        const float *token_type,
                                                        const float *gamma,
                                                        const float *beta,
                                                        int total,
                                                        int hidden,
                                                        int vocab,
                                                        float eps) {
    extern __shared__ float rowbuf[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= total)
        return;
    int id = ids[row];
    int pos = positions[row];

    float s = 0.0f;
    for (int d = tid; d < hidden; d += blockDim.x) {
        float tok = 0.0f;
        if (id >= 0 && id < vocab)
            tok = load_embedding_value(emb, emb_dtype, (size_t)id * hidden + d);
        float v = tok + (pos_emb[(size_t)pos * hidden + d] + token_type[d]);
        rowbuf[d] = v;
        s += v;
    }
    float mean = block_sum(s) / (float)hidden;
    float vs = 0.0f;
    for (int d = tid; d < hidden; d += blockDim.x) {
        float c = rowbuf[d] - mean;
        vs += c * c;
    }
    float inv = rsqrtf(block_sum(vs) / (float)hidden + eps);
    __half *o = out + (size_t)row * hidden;
    for (int d = tid; d < hidden; d += blockDim.x) {
        float nv = (rowbuf[d] - mean) * inv * gamma[d] + beta[d];
        o[d] = __float2half(nv);
    }
}

// BERT SkipLayerNorm variant for the common post-GEMM shape. The residual has
// already been accumulated by cuBLAS beta=1, so this fuses the remaining row
// bias with LayerNorm and keeps the pre-normalized row in shared memory.
__global__ static void layer_norm_bias_kernel(float *out,
                                              __nv_bfloat16 *out_bf16,
                                              const float *x,
                                              const float *bias,
                                              const float *gamma,
                                              const float *beta,
                                              int rows,
                                              int dim,
                                              float eps) {
    extern __shared__ float rowbuf[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows)
        return;
    const float *xr = x + (size_t)row * dim;
    float s = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = xr[d] + bias[d];
        rowbuf[d] = v;
        s += v;
    }
    float mean = block_sum(s) / (float)dim;
    float vs = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float c = rowbuf[d] - mean;
        vs += c * c;
    }
    float inv = rsqrtf(block_sum(vs) / (float)dim + eps);
    float *o = out + (size_t)row * dim;
    // Post-norm BERT: the LayerNorm output is both the F32 residual carried to
    // the next add and the input to the next GEMM. The optional bf16 copy lets
    // the bf16-activation path feed the GEMM directly (no per-GEMM F32->bf16
    // cast) while the residual stays F32 for the mean-subtracting norm.
    __nv_bfloat16 *ob = out_bf16 ? out_bf16 + (size_t)row * dim : nullptr;
    for (int d = tid; d < dim; d += blockDim.x) {
        float nv = (rowbuf[d] - mean) * inv * gamma[d] + beta[d];
        o[d] = nv;
        if (ob)
            ob[d] = __float2bfloat16(nv);
    }
}

// BERT reduced-precision residual stream: the projection GEMM writes only its
// F32 output to `x`; this kernel widens the previous BF16 residual, applies
// residual + bias + LayerNorm in F32, and stores the normalized residual as BF16.
__global__ static void layer_norm_bias_resid_bf16_kernel(__nv_bfloat16 *out,
                                                         const __nv_bfloat16 *resid,
                                                         const float *x,
                                                         const float *bias,
                                                         const float *gamma,
                                                         const float *beta,
                                                         int rows,
                                                         int dim,
                                                         float eps) {
    extern __shared__ float rowbuf[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows)
        return;
    const float *xr = x + (size_t)row * dim;
    const __nv_bfloat16 *rr = resid + (size_t)row * dim;
    float s = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = xr[d] + __bfloat162float(rr[d]) + bias[d];
        rowbuf[d] = v;
        s += v;
    }
    float mean = block_sum(s) / (float)dim;
    float vs = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float c = rowbuf[d] - mean;
        vs += c * c;
    }
    float inv = rsqrtf(block_sum(vs) / (float)dim + eps);
    __nv_bfloat16 *o = out + (size_t)row * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        float nv = (rowbuf[d] - mean) * inv * gamma[d] + beta[d];
        o[d] = __float2bfloat16(nv);
    }
}

__global__ static void layer_norm_bias_f16_kernel(float *out,
                                                  __half *out_f16,
                                                  const float *x,
                                                  const float *bias,
                                                  const float *gamma,
                                                  const float *beta,
                                                  int rows,
                                                  int dim,
                                                  float eps) {
    extern __shared__ float rowbuf[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows)
        return;
    const float *xr = x + (size_t)row * dim;
    float s = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = xr[d] + bias[d];
        rowbuf[d] = v;
        s += v;
    }
    float mean = block_sum(s) / (float)dim;
    float vs = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float c = rowbuf[d] - mean;
        vs += c * c;
    }
    float inv = rsqrtf(block_sum(vs) / (float)dim + eps);
    float *o = out + (size_t)row * dim;
    __half *oh = out_f16 ? out_f16 + (size_t)row * dim : nullptr;
    for (int d = tid; d < dim; d += blockDim.x) {
        float nv = (rowbuf[d] - mean) * inv * gamma[d] + beta[d];
        o[d] = nv;
        if (oh)
            oh[d] = __float2half(nv);
    }
}

__global__ static void layer_norm_bias_resid_f16_kernel(__half *out,
                                                        const __half *resid,
                                                        const float *x,
                                                        const float *bias,
                                                        const float *gamma,
                                                        const float *beta,
                                                        int rows,
                                                        int dim,
                                                        float eps) {
    extern __shared__ float rowbuf[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows)
        return;
    const float *xr = x + (size_t)row * dim;
    const __half *rr = resid + (size_t)row * dim;
    float s = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = xr[d] + __half2float(rr[d]) + bias[d];
        rowbuf[d] = v;
        s += v;
    }
    float mean = block_sum(s) / (float)dim;
    float vs = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float c = rowbuf[d] - mean;
        vs += c * c;
    }
    float inv = rsqrtf(block_sum(vs) / (float)dim + eps);
    __half *o = out + (size_t)row * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        float nv = (rowbuf[d] - mean) * inv * gamma[d] + beta[d];
        o[d] = __float2half(nv);
    }
}

// Reduced-precision residual delta: identical reductions to the *_resid_*
// kernels above, but the freshly-projected delta (wo / down output) is read as
// the active 16-bit dtype instead of F32. The residual add and the LayerNorm
// mean/variance stay in F32; only the projection output is rounded, which
// removes one F32 [rows x dim] write+reread per attention/FFN block. Active with
// the 16-bit residual stream (FFWD_CUDA_BERT_DELTA16=0 forces the F32 delta back
// for A/B); validated at parity, cosine vs F32 ~0.99998 on bge-base.
__global__ static void layer_norm_bias_resid_bf16_delta16_kernel(__nv_bfloat16 *out,
                                                                 const __nv_bfloat16 *resid,
                                                                 const __nv_bfloat16 *delta,
                                                                 const float *bias,
                                                                 const float *gamma,
                                                                 const float *beta,
                                                                 int rows,
                                                                 int dim,
                                                                 float eps) {
    extern __shared__ float rowbuf[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows)
        return;
    const __nv_bfloat16 *dr = delta + (size_t)row * dim;
    const __nv_bfloat16 *rr = resid + (size_t)row * dim;
    float s = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = __bfloat162float(dr[d]) + __bfloat162float(rr[d]) + bias[d];
        rowbuf[d] = v;
        s += v;
    }
    float mean = block_sum(s) / (float)dim;
    float vs = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float c = rowbuf[d] - mean;
        vs += c * c;
    }
    float inv = rsqrtf(block_sum(vs) / (float)dim + eps);
    __nv_bfloat16 *o = out + (size_t)row * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        float nv = (rowbuf[d] - mean) * inv * gamma[d] + beta[d];
        o[d] = __float2bfloat16(nv);
    }
}

__global__ static void layer_norm_bias_resid_f16_delta16_kernel(__half *out,
                                                                const __half *resid,
                                                                const __half *delta,
                                                                const float *bias,
                                                                const float *gamma,
                                                                const float *beta,
                                                                int rows,
                                                                int dim,
                                                                float eps) {
    extern __shared__ float rowbuf[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows)
        return;
    const __half *dr = delta + (size_t)row * dim;
    const __half *rr = resid + (size_t)row * dim;
    float s = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = __half2float(dr[d]) + __half2float(rr[d]) + bias[d];
        rowbuf[d] = v;
        s += v;
    }
    float mean = block_sum(s) / (float)dim;
    float vs = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float c = rowbuf[d] - mean;
        vs += c * c;
    }
    float inv = rsqrtf(block_sum(vs) / (float)dim + eps);
    __half *o = out + (size_t)row * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        float nv = (rowbuf[d] - mean) * inv * gamma[d] + beta[d];
        o[d] = __float2half(nv);
    }
}

__global__ static void bias_gelu_kernel(float *x, const float *bias, int rows, int dim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * dim;
    if (idx < n) {
        int d = (int)(idx % dim);
        float v = x[idx] + bias[d];
        x[idx] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
    }
}

// bf16-output GELU variants for the bf16-activation BERT path: read the F32 FFN
// intermediate, apply bias+GELU, write bf16 for the down-projection GEMM input.
__global__ static void
bias_gelu_to_bf16_kernel(__nv_bfloat16 *out, const float *x, const float *bias, int rows, int dim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * dim;
    if (idx < n) {
        int d = (int)(idx % dim);
        float v = x[idx] + bias[d];
        out[idx] = __float2bfloat16(0.5f * v * (1.0f + erff(v * 0.70710678118654752f)));
    }
}

__global__ static void
bias_gelu_to_f16_kernel(__half *out, const float *x, const float *bias, int rows, int dim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * dim;
    if (idx < n) {
        int d = (int)(idx % dim);
        float v = x[idx] + bias[d];
        out[idx] = __float2half(0.5f * v * (1.0f + erff(v * 0.70710678118654752f)));
    }
}

template <typename IN_T, typename OUT_T>
__global__ static void
bias_gelu_16_to_16_kernel(OUT_T *out, const IN_T *x, const float *bias, int rows, int dim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * dim;
    if (idx < n) {
        int d = (int)(idx % dim);
        float v = load_act(x, idx) + bias[d];
        store_act(out, idx, 0.5f * v * (1.0f + erff(v * 0.70710678118654752f)));
    }
}

__global__ static void bias_gelu_tanh_kernel(float *x, const float *bias, int rows, int dim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * dim;
    if (idx < n) {
        int d = (int)(idx % dim);
        float v = x[idx] + bias[d];
        float inner = 0.79788456080286536f * (v + 0.044715f * v * v * v);
        x[idx] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

__global__ static void bias_gelu_tanh_to_bf16_kernel(
    __nv_bfloat16 *out, const float *x, const float *bias, int rows, int dim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * dim;
    if (idx < n) {
        int d = (int)(idx % dim);
        float v = x[idx] + bias[d];
        float inner = 0.79788456080286536f * (v + 0.044715f * v * v * v);
        out[idx] = __float2bfloat16(0.5f * v * (1.0f + tanhf(inner)));
    }
}

__global__ static void
bias_gelu_tanh_to_f16_kernel(__half *out, const float *x, const float *bias, int rows, int dim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * dim;
    if (idx < n) {
        int d = (int)(idx % dim);
        float v = x[idx] + bias[d];
        float inner = 0.79788456080286536f * (v + 0.044715f * v * v * v);
        out[idx] = __float2half(0.5f * v * (1.0f + tanhf(inner)));
    }
}

template <typename IN_T, typename OUT_T>
__global__ static void
bias_gelu_tanh_16_to_16_kernel(OUT_T *out, const IN_T *x, const float *bias, int rows, int dim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)rows * dim;
    if (idx < n) {
        int d = (int)(idx % dim);
        float v = load_act(x, idx) + bias[d];
        float inner = 0.79788456080286536f * (v + 0.044715f * v * v * v);
        store_act(out, idx, 0.5f * v * (1.0f + tanhf(inner)));
    }
}

static int launch_check(void) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "cuda kernel failed: %s\n", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

// Compute type for CUDA GEMMs. Default is exact F32 (== cublasSgemm).
// FFWD_CUDA_GEMM_MODE={bf16,tf32,16f} selects a reduced-precision tensor-core
// matmul that keeps F32 inputs/outputs and F32 accumulation but rounds operands
// internally.
static cublasComputeType_t gemm_compute(void) {
    static int env_init = 0;
    if (!g_gemm_compute_set && !env_init) {
        const char *e = getenv("FFWD_CUDA_GEMM_MODE");
        if (e && parse_gemm_compute(e, &g_gemm_compute) == 0)
            g_gemm_compute_set = 1;
        env_init = 1;
    }
    return g_gemm_compute;
}

// Streaming attention selector. The default path runs the promoted tc3
// tensor-core flash kernel for Qwen BF16 HD=128 and the materialized cuBLAS path
// for everything else. FFWD_CUDA_STREAMING_ATTN=mat/off/materialized forces the
// Qwen BF16 path back to materialized for A/B; =tc3/mma_kv_ldm/mma_ldmatrix
// explicitly selects the same promoted kernel.
static int streaming_attention_mode(void) {
    static int init = 0;
    static int mode = 0; // -1 force materialized, 0 default, 8 explicit tc3
    if (!init) {
        const char *e = getenv("FFWD_CUDA_STREAMING_ATTN");
        if (e && (!strcmp(e, "tc3") || !strcmp(e, "mma_kv_ldm") || !strcmp(e, "mma_ldmatrix")))
            mode = 8;
        else if (e && (!strcmp(e, "mat") || !strcmp(e, "off") || !strcmp(e, "materialized")))
            mode = -1;
        init = 1;
    }
    return mode;
}

// Query tiles per block for the K/V-sharing kernels. FFWD_TC_WARPS overrides
// the mode default; valid values are 2/4/8.
static int tc_kv_warps_env(void) {
    static int init = 0, w = 0;
    if (!init) {
        const char *e = getenv("FFWD_TC_WARPS");
        if (e) {
            int v = atoi(e);
            if (v == 2 || v == 4 || v == 8)
                w = v;
        }
        init = 1;
    }
    return w;
}

static int tc_kv_warps_or(int default_w) {
    int w = tc_kv_warps_env();
    return w ? w : default_w;
}

static int use_streaming_attention(const ffwd_config_t *c, int out_bf16, int head_dim) {
    int mode = streaming_attention_mode();
    if (mode < 0) // FFWD_CUDA_STREAMING_ATTN=mat/off: force the materialized path
        return 0;
    // Promoted default (no FFWD_CUDA_STREAMING_ATTN set): the Qwen BF16 HD=128
    // path runs the tc3 tensor-core flash kernel - faster than materialized BF16
    // attention (1.2-2.4x, growing with length) at parity cosine ~0.99997. Every
    // other default case (F32, BERT, non-128 head_dim) keeps the materialized
    // path. Explicit tc3 uses the same predicate.
    return (mode == 0 || mode == 8) && c->family == FFWD_FAMILY_QWEN3 && out_bf16 && head_dim == 128;
}

// Grow the F32 weight-widen scratch to hold `elems` floats. The memory-only
// BF16 path materializes one F32 weight at a time into this buffer.
static int ensure_weight_f32(ffwd_cuda_ctx_t *ctx, size_t elems) {
    if (ctx->weight_f32_elems >= elems)
        return 0;
    cudaFree(ctx->weight_f32);
    ctx->weight_f32 = NULL;
    ctx->weight_f32_elems = 0;
    CUDA_CHECK(cudaMalloc((void **)&ctx->weight_f32, elems * sizeof(float)));
    ctx->weight_f32_elems = elems;
    return 0;
}

// y = w @ x (+ beta*y). With 16-bit weights and already-16-bit activations, both
// inputs are read in that storage dtype with F32 accumulation. For BERT in an
// exact-or-TF32 compute mode, a F32 activation with reduced-precision weights is
// handled by widening the weight back to F32 first: reduced precision is then a
// storage choice only. In explicit reduced-compute modes (bf16/16f), exactness is
// already given up, so we cast F32 activations to the matching 16-bit operand and
// let tensor cores do the GEMM.
static int linear_ex(ffwd_cuda_ctx_t *ctx,
                     const cuda_matrix_t *w,
                     const void *x,
                     cuda_dtype_t x_dtype,
                     float *y,
                     int rows,
                     int in_dim,
                     int out_dim,
                     int x_stride,
                     float beta) {
    const float alpha = 1.0f;
    if (w->dtype == CUDA_DTYPE_BF16 || w->dtype == CUDA_DTYPE_F16) {
        cudaDataType input_type = w->dtype == CUDA_DTYPE_BF16 ? CUDA_R_16BF : CUDA_R_16F;
        // 16-bit float compute (bf16/16f) reads bf16 weights directly; exact F32
        // and TF32 need F32 weight operands, so BERT widens its bf16 weight then.
        cublasComputeType_t gc = gemm_compute();
        int compute_16bit = (gc == CUBLAS_COMPUTE_32F_FAST_16BF || gc == CUBLAS_COMPUTE_32F_FAST_16F);
        if (x_dtype == CUDA_DTYPE_F32 && ctx->config.family == FFWD_FAMILY_BERT && !compute_16bit) {
            size_t welems = (size_t)out_dim * (size_t)in_dim;
            if (ensure_weight_f32(ctx, welems) != 0)
                return -1;
            int wt = 256;
            if (w->dtype == CUDA_DTYPE_BF16)
                cast_bf16_to_f32_kernel<<<(welems + wt - 1) / wt, wt, 0, ctx->stream>>>(
                    ctx->weight_f32, (const __nv_bfloat16 *)w->d, welems);
            else
                cast_f16_to_f32_kernel<<<(welems + wt - 1) / wt, wt, 0, ctx->stream>>>(
                    ctx->weight_f32, (const __half *)w->d, welems);
            if (launch_check() != 0)
                return -1;
            CUBLAS_CHECK(cublasGemmEx(ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N, out_dim, rows, in_dim,
                                      &alpha, ctx->weight_f32, CUDA_R_32F, in_dim, x, CUDA_R_32F,
                                      x_stride, &beta, y, CUDA_R_32F, out_dim, gc,
                                      CUBLAS_GEMM_DEFAULT));
            return 0;
        }
        const void *xb = x;
        if (x_dtype == CUDA_DTYPE_F32) {
            size_t n = (size_t)rows * (size_t)x_stride;
            int threads = 256;
            if (w->dtype == CUDA_DTYPE_BF16) {
                cast_f32_to_bf16_kernel<<<(n + threads - 1) / threads, threads, 0, ctx->stream>>>(
                    (__nv_bfloat16 *)ctx->act_bf16, (const float *)x, n);
                xb = ctx->act_bf16;
            } else {
                cast_f32_to_f16_kernel<<<(n + threads - 1) / threads, threads, 0, ctx->stream>>>(
                    (__half *)ctx->act_f16, (const float *)x, n);
                xb = ctx->act_f16;
            }
            if (launch_check() != 0)
                return -1;
        } else if (x_dtype != w->dtype) {
            fprintf(stderr, "cuda: mixed 16-bit GEMM operand dtypes are unsupported\n");
            return -1;
        }
        CUBLAS_CHECK(cublasGemmEx(ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N, out_dim, rows, in_dim, &alpha,
                                  w->d, input_type, in_dim, xb, input_type, x_stride, &beta, y,
                                  CUDA_R_32F, out_dim, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        return 0;
    }
    CUBLAS_CHECK(cublasGemmEx(ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N, out_dim, rows, in_dim, &alpha,
                              w->d, CUDA_R_32F, in_dim, x, CUDA_R_32F, x_stride, &beta, y, CUDA_R_32F,
                              out_dim, gemm_compute(), CUBLAS_GEMM_DEFAULT));
    return 0;
}

static int linear(ffwd_cuda_ctx_t *ctx,
                  const cuda_matrix_t *w,
                  const float *x,
                  float *y,
                  int rows,
                  int in_dim,
                  int out_dim) {
    return linear_ex(ctx, w, x, CUDA_DTYPE_F32, y, rows, in_dim, out_dim, in_dim, 0.0f);
}

// x points at activations already stored as BF16 (e.g. ctx->act_bf16).
static int linear_bf16x(ffwd_cuda_ctx_t *ctx,
                        const cuda_matrix_t *w,
                        const void *x,
                        float *y,
                        int rows,
                        int in_dim,
                        int out_dim,
                        int x_stride,
                        float beta) {
    return linear_ex(ctx, w, x, CUDA_DTYPE_BF16, y, rows, in_dim, out_dim, x_stride, beta);
}

// x points at activations already stored as FP16 (e.g. ctx->act_f16).
static int linear_f16x(ffwd_cuda_ctx_t *ctx,
                       const cuda_matrix_t *w,
                       const void *x,
                       float *y,
                       int rows,
                       int in_dim,
                       int out_dim,
                       int x_stride,
                       float beta) {
    return linear_ex(ctx, w, x, CUDA_DTYPE_F16, y, rows, in_dim, out_dim, x_stride, beta);
}

static int linear_16x_to_16(ffwd_cuda_ctx_t *ctx,
                            const cuda_matrix_t *w,
                            const void *x,
                            cuda_dtype_t x_dtype,
                            void *y,
                            cuda_dtype_t y_dtype,
                            int rows,
                            int in_dim,
                            int out_dim,
                            int x_stride) {
    if ((y_dtype != CUDA_DTYPE_BF16 && y_dtype != CUDA_DTYPE_F16) || x_dtype != y_dtype ||
        w->dtype != y_dtype) {
        fprintf(stderr, "cuda: linear_16x_to_16 requires matching 16-bit operand dtypes\n");
        return -1;
    }
    const float alpha = 1.0f, beta = 0.0f;
    cudaDataType t = y_dtype == CUDA_DTYPE_BF16 ? CUDA_R_16BF : CUDA_R_16F;
    CUBLAS_CHECK(cublasGemmEx(ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N, out_dim, rows, in_dim, &alpha,
                              w->d, t, in_dim, x, t, x_stride, &beta, y, t, out_dim,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    return 0;
}

static int linear_accum(ffwd_cuda_ctx_t *ctx,
                        const cuda_matrix_t *w,
                        const float *x,
                        float *y,
                        int rows,
                        int in_dim,
                        int out_dim) {
    return linear_ex(ctx, w, x, CUDA_DTYPE_F32, y, rows, in_dim, out_dim, in_dim, 1.0f);
}

static void linear_lt_plan_free(ffwd_cuda_ctx_t *ctx) {
    if (!ctx->lt_have)
        return;
    cublasLtMatrixLayoutDestroy(ctx->lt_D);
    cublasLtMatrixLayoutDestroy(ctx->lt_B);
    cublasLtMatrixLayoutDestroy(ctx->lt_A);
    cublasLtMatmulDescDestroy(ctx->lt_desc);
    ctx->lt_have = 0;
}

// Fused projection via cuBLASLt: y = epilogue(W^T @ x). The epilogue applies the
// row bias and, for CUBLASLT_EPILOGUE_GELU_BIAS, the exact erf GeLU inside the
// GEMM, so a separate bias/activation kernel and its [out_dim x rows] HBM
// round-trip are removed. Operands are 16-bit (bf16/f16); the output is bf16/f16
// (FFN gate-up) or F32 (qkv projection), accumulation is F32. The matmul plan
// (descriptor + layouts + algo) is cached on the context keyed by shape, so the
// heuristic runs ~once per request rather than once per layer (a single slot, so
// the qkv and gate-up shapes alternate-evict, which is still correct and cheap).
static int linear_lt(ffwd_cuda_ctx_t *ctx,
                     const cuda_matrix_t *w,
                     const void *x,
                     void *y,
                     cuda_dtype_t out_dtype,
                     const float *bias,
                     cublasLtEpilogue_t epilogue,
                     int rows,
                     int in_dim,
                     int out_dim,
                     int x_stride) {
    if (w->dtype != CUDA_DTYPE_BF16 && w->dtype != CUDA_DTYPE_F16) {
        fprintf(stderr, "cuda: linear_lt requires 16-bit weights\n");
        return -1;
    }
    cudaDataType io = w->dtype == CUDA_DTYPE_BF16 ? CUDA_R_16BF : CUDA_R_16F;
    cudaDataType od = out_dtype == CUDA_DTYPE_F16
                          ? CUDA_R_16F
                          : (out_dtype == CUDA_DTYPE_BF16 ? CUDA_R_16BF : CUDA_R_32F);
    const float alpha = 1.0f, beta = 0.0f;

    // The bias dtype must match the output (cuBLASLt rejects an F32 bias on a
    // 16-bit output): for a 16-bit output cast the F32 bias into lt_bias16; for an
    // F32 output (the qkv projection) use the F32 bias directly. The bias POINTER
    // is set per call below (it varies per layer for the F32 path), not baked into
    // the cached plan.
    const void *bias_dev;
    cudaDataType bias_t;
    if (od == CUDA_R_16BF) {
        cast_f32_to_bf16_kernel<<<(out_dim + 255) / 256, 256, 0, ctx->stream>>>(
            (__nv_bfloat16 *)ctx->lt_bias16, bias, out_dim);
        bias_dev = ctx->lt_bias16;
        bias_t = CUDA_R_16BF;
    } else if (od == CUDA_R_16F) {
        cast_f32_to_f16_kernel<<<(out_dim + 255) / 256, 256, 0, ctx->stream>>>(
            (__half *)ctx->lt_bias16, bias, out_dim);
        bias_dev = ctx->lt_bias16;
        bias_t = CUDA_R_16F;
    } else {
        bias_dev = bias;
        bias_t = CUDA_R_32F;
    }

    int key_ok = ctx->lt_have && ctx->lt_k_rows == rows && ctx->lt_k_in == in_dim &&
                 ctx->lt_k_out == out_dim && ctx->lt_k_xs == x_stride &&
                 ctx->lt_k_epi == (int)epilogue && ctx->lt_k_io == (int)io && ctx->lt_k_od == (int)od;
    if (!key_ok) {
        linear_lt_plan_free(ctx);
        cublasLtMatmulDesc_t op = NULL;
        cublasLtMatrixLayout_t A = NULL, B = NULL, D = NULL;
        cublasLtMatmulPreference_t pref = NULL;
        int ok = 0;
        do {
            if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
                CUBLAS_STATUS_SUCCESS)
                break;
            cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                           sizeof(epilogue));
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_t,
                                           sizeof(bias_t));
            // A = weight stored [in_dim x out_dim] col-major (ld in_dim), op transposes it;
            // B = x stored [in_dim x rows] (ld x_stride); D = y stored [out_dim x rows].
            if (cublasLtMatrixLayoutCreate(&A, io, in_dim, out_dim, in_dim) != CUBLAS_STATUS_SUCCESS)
                break;
            if (cublasLtMatrixLayoutCreate(&B, io, in_dim, rows, x_stride) != CUBLAS_STATUS_SUCCESS)
                break;
            if (cublasLtMatrixLayoutCreate(&D, od, out_dim, rows, out_dim) != CUBLAS_STATUS_SUCCESS)
                break;
            if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS)
                break;
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &ctx->lt_ws_bytes, sizeof(ctx->lt_ws_bytes));
            int got = 0;
            cublasStatus_t hst =
                cublasLtMatmulAlgoGetHeuristic(ctx->lt, op, A, B, D, D, pref, 1, &ctx->lt_heur, &got);
            if (hst != CUBLAS_STATUS_SUCCESS || got == 0) {
                fprintf(stderr, "cuda: linear_lt heuristic st=%d got=%d (m=%d n=%d k=%d epi=%d)\n",
                        (int)hst, got, out_dim, rows, in_dim, (int)epilogue);
                break;
            }
            ok = 1;
        } while (0);
        if (pref)
            cublasLtMatmulPreferenceDestroy(pref);
        if (!ok) {
            if (D)
                cublasLtMatrixLayoutDestroy(D);
            if (B)
                cublasLtMatrixLayoutDestroy(B);
            if (A)
                cublasLtMatrixLayoutDestroy(A);
            if (op)
                cublasLtMatmulDescDestroy(op);
            fprintf(stderr, "cuda: linear_lt plan build failed\n");
            return -1;
        }
        ctx->lt_desc = op;
        ctx->lt_A = A;
        ctx->lt_B = B;
        ctx->lt_D = D;
        ctx->lt_have = 1;
        ctx->lt_k_rows = rows;
        ctx->lt_k_in = in_dim;
        ctx->lt_k_out = out_dim;
        ctx->lt_k_xs = x_stride;
        ctx->lt_k_epi = (int)epilogue;
        ctx->lt_k_io = (int)io;
        ctx->lt_k_od = (int)od;
    }
    // The bias pointer is not baked into the cached plan (it varies per layer for
    // the F32 qkv path), so set it on the descriptor every call.
    cublasLtMatmulDescSetAttribute(ctx->lt_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_dev,
                                   sizeof(bias_dev));
    cublasStatus_t mst = cublasLtMatmul(ctx->lt, ctx->lt_desc, &alpha, w->d, ctx->lt_A, x, ctx->lt_B,
                                        &beta, y, ctx->lt_D, y, ctx->lt_D, &ctx->lt_heur.algo,
                                        ctx->lt_workspace, ctx->lt_ws_bytes, ctx->stream);
    if (mst != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuda: linear_lt matmul st=%d\n", (int)mst);
        return -1;
    }
    return 0;
}

static int gemm_strided_batched(cublasHandle_t blas,
                                cublasOperation_t transa,
                                cublasOperation_t transb,
                                int m,
                                int n,
                                int k,
                                const float *alpha,
                                const float *A,
                                int lda,
                                long long strideA,
                                const float *B,
                                int ldb,
                                long long strideB,
                                const float *beta,
                                float *C,
                                int ldc,
                                long long strideC,
                                int batch_count) {
    cublasComputeType_t ct = gemm_compute();
    if (ct == CUBLAS_COMPUTE_32F) {
        CUBLAS_CHECK(cublasSgemmStridedBatched(blas, transa, transb, m, n, k, alpha, A, lda, strideA,
                                               B, ldb, strideB, beta, C, ldc, strideC, batch_count));
    } else {
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            blas, transa, transb, m, n, k, alpha, A, CUDA_R_32F, lda, strideA, B, CUDA_R_32F, ldb,
            strideB, beta, C, CUDA_R_32F, ldc, strideC, batch_count, ct, CUBLAS_GEMM_DEFAULT));
    }
    return 0;
}

// Attention for one micro-batch. Expands K/V to the per-query-head layout,
// then for each sequence runs batched-over-heads Q@K^T (scaled) -> row softmax
// -> @V via cuBLAS. Requires ctx->kexp, ctx->vexp ([total x q_dim]) and
// ctx->attn_scores ([n_heads x max_seq x max_seq]) to be allocated. h_offsets
// is the host copy of the packed sequence boundaries.
// Prepare the equal-length batched-attention path for one forward pass:
// grow the score/probability scratch to hold attn_G sequences at once and
// upload the pointer arrays for the two batched GEMMs (buffer addresses are
// layer-invariant). attn_G stays 0 - per-sequence fallback - for ragged
// batches, single sequences, or when one sequence's scores exceed the
// budget.
// use_bf16 must match the out_bf16 branch the caller will take in
// cuda_attention_gemm: the V/probability/output pointer layout chosen here has to
// agree with the data type that GEMM reads. The materialized BERT fallback runs
// attention in F32 (passes out_bf16=NULL) even with BF16 weights, so it must pass
// use_bf16=0; the Qwen path runs BF16 attention when its context's weights are
// BF16. This must stay a caller-owned decision; deriving it inside this helper
// previously mismatched the BERT path (BF16 layout, F32 GEMM) and corrupted
// equal-length batches.
static int attn_batched_setup(
    ffwd_cuda_ctx_t *ctx, int batch, int q_offset, int qkv_dim, cuda_dtype_t attn_dtype) {
    const ffwd_config_t *c = &ctx->config;
    ctx->attn_G = 0;
    if (batch < 2)
        return 0;
    int L = ctx->offsets_host[1] - ctx->offsets_host[0];
    if (L > CUDA_ATTN_BATCHED_MAX_LEN)
        return 0;
    for (int b = 1; b < batch; b++)
        if (ctx->offsets_host[b + 1] - ctx->offsets_host[b] != L)
            return 0;
    int H = c->n_heads, hd = c->head_dim, q_dim = c->q_dim;
    long long per = (long long)H * L * L;
    if (per <= 0 || per > CUDA_ATTN_SCORES_BUDGET / 2)
        return 0;
    int G = (int)(CUDA_ATTN_SCORES_BUDGET / per);
    if (G > batch)
        G = batch;
    if (G < 2)
        return 0;

    int use_16 = attn_dtype == CUDA_DTYPE_BF16 || attn_dtype == CUDA_DTYPE_F16;
    long long want = (long long)G * per;
    if (want > ctx->attn_scores_elems) {
        cudaFree(ctx->attn_scores);
        ctx->attn_scores = NULL;
        ctx->attn_scores_elems = 0;
        CUDA_CHECK(cudaMalloc((void **)&ctx->attn_scores, (size_t)want * sizeof(float)));
        ctx->attn_scores_elems = want;
    }
    if (use_16 && want > ctx->attn_probs_elems) {
        cudaFree(ctx->attn_probs);
        ctx->attn_probs = NULL;
        ctx->attn_probs_elems = 0;
        CUDA_CHECK(cudaMalloc(&ctx->attn_probs, (size_t)want * sizeof(uint16_t)));
        ctx->attn_probs_elems = want;
    }

    int entries = 6 * batch * H;
    if (entries > ctx->attn_ptrs_cap) {
        cudaFree((void *)ctx->attn_ptrs);
        free((void *)ctx->attn_ptrs_host);
        ctx->attn_ptrs = NULL;
        ctx->attn_ptrs_cap = 0;
        ctx->attn_ptrs_host = (const void **)malloc((size_t)entries * sizeof(void *));
        if (!ctx->attn_ptrs_host)
            return -1;
        CUDA_CHECK(cudaMalloc((void **)&ctx->attn_ptrs, (size_t)entries * sizeof(void *)));
        ctx->attn_ptrs_cap = entries;
    }

    const void **K = ctx->attn_ptrs_host;
    const void **Q = K + (size_t)batch * H;
    const void **S = Q + (size_t)batch * H;
    const void **V = S + (size_t)batch * H;
    const void **P = V + (size_t)batch * H;
    const void **O = P + (size_t)batch * H;
    const __nv_bfloat16 *vb = (const __nv_bfloat16 *)ctx->vexp;
    const __nv_bfloat16 *pb = (const __nv_bfloat16 *)ctx->attn_probs;
    const __nv_bfloat16 *ab = (const __nv_bfloat16 *)ctx->act_bf16;
    const __half *vh = (const __half *)ctx->vexp;
    const __half *ph = (const __half *)ctx->attn_probs;
    const __half *ah = (const __half *)ctx->act_f16;
    for (int b = 0; b < batch; b++) {
        // Score/probability rows cycle every G sequences (chunked reuse).
        long long srow = (long long)(b % G) * per;
        size_t tok = (size_t)b * L;
        for (int h = 0; h < H; h++) {
            size_t i = (size_t)b * H + h;
            K[i] = ctx->kexp + tok * q_dim + (size_t)h * hd;
            Q[i] = ctx->qkv + tok * qkv_dim + q_offset + (size_t)h * hd;
            S[i] = ctx->attn_scores + srow + (long long)h * L * L;
            if (attn_dtype == CUDA_DTYPE_BF16) {
                V[i] = vb + tok * q_dim + (size_t)h * hd;
                P[i] = pb + srow + (long long)h * L * L;
                O[i] = ab + tok * q_dim + (size_t)h * hd;
            } else if (attn_dtype == CUDA_DTYPE_F16) {
                V[i] = vh + tok * q_dim + (size_t)h * hd;
                P[i] = ph + srow + (long long)h * L * L;
                O[i] = ah + tok * q_dim + (size_t)h * hd;
            } else {
                V[i] = ctx->vexp + tok * q_dim + (size_t)h * hd;
                P[i] = S[i];
                O[i] = ctx->attn_out + tok * q_dim + (size_t)h * hd;
            }
        }
    }
    CUDA_CHECK(cudaMemcpy((void *)ctx->attn_ptrs, ctx->attn_ptrs_host,
                          (size_t)entries * sizeof(void *), cudaMemcpyHostToDevice));
    ctx->attn_G = G;
    ctx->attn_L = L;
    return 0;
}

// BF16 tensor-core flash attention for the BERT shapes (head_dim 32/64,
// bidirectional), replacing the materialized N×N scores/probs path. It is bit-
// accurate (cosine vs F32 >= the materialized bf16 path). It is the DEFAULT on
// the bf16w-bf16 head_dim-64 path, where the O(n^2) HBM removal wins at every
// measured length. The head_dim-32 MiniLM shape is implemented but opt-in
// (FFWD_CUDA_BERT_FLASH=1), because H100 validation did not show a reliable
// default win. FFWD_CUDA_BERT_FLASH=0 forces the materialized path back for A/B.
// FFWD_TC_WARPS (2/4/8) tunes the query tiles per block (default 8).
static int bert_flash_mode(void) {
    static int init = 0, mode = 0;
    if (!init) {
        const char *e = getenv("FFWD_CUDA_BERT_FLASH");
        if (e && (!strcmp(e, "0") || !strcmp(e, "off")))
            mode = -1;
        else if (e &&
                 (!strcmp(e, "1") || !strcmp(e, "on") || !strcmp(e, "force") || !strcmp(e, "all")))
            mode = 1;
        init = 1;
    }
    return mode;
}

// BERT flash K/V staging. Unset/default uses direct packed-QKV K/V loads whenever
// QKV16 is active; 1/on keeps that direct path enabled; 0/off keeps the separate
// K/V expansion for A/B. This rides QKV16 and never applies to F32 qkv.
static int bert_direct_kv_mode(void) {
    static int init = 0, mode = 0;
    if (!init) {
        const char *e = getenv("FFWD_CUDA_BERT_DIRECT_KV");
        if (e && (!strcmp(e, "1") || !strcmp(e, "on") || !strcmp(e, "force")))
            mode = 1;
        else if (e && (!strcmp(e, "0") || !strcmp(e, "off")))
            mode = -1;
        init = 1;
    }
    return mode;
}

// out_16 != NULL selects a reduced-precision attention layout: V and the softmax
// probabilities are stored in out_dtype, and the @V stage emits attention output
// in the same dtype, ready for the output projection without a cast.
static int cuda_attention_gemm(ffwd_cuda_ctx_t *ctx,
                               const int *h_offsets,
                               int batch,
                               int q_offset,
                               int k_offset,
                               int v_offset,
                               int qkv_dim,
                               const void *qkv_src,
                               cuda_dtype_t qkv_dtype,
                               float scale,
                               void *out_16,
                               cuda_dtype_t out_dtype,
                               int bert_flash_pipeline) {
    const ffwd_config_t *c = &ctx->config;
    int q_dim = c->q_dim;
    int hd = c->head_dim;
    int H = c->n_heads;
    int total = h_offsets[batch];
    int max_L = 0;
    for (int b = 0; b < batch; b++) {
        int L = h_offsets[b + 1] - h_offsets[b];
        if (L > max_L)
            max_L = L;
    }
    int causal = c->attention_mode == FFWD_ATTENTION_CAUSAL;
    const float alpha = scale, beta = 0.0f, one = 1.0f;
    int stream_mode = streaming_attention_mode();
    int use_stream = use_streaming_attention(c, out_dtype == CUDA_DTYPE_BF16, hd);
    int use_tc_kexp_bf16 = use_stream && out_dtype == CUDA_DTYPE_BF16 &&
                           c->family == FFWD_FAMILY_QWEN3 && hd == 128 &&
                           (stream_mode == 0 || stream_mode == 8);
    // BERT flash attention: head_dim 32/64, bidirectional, K/V expanded in the
    // requested reduced dtype.
    int bert_flash = bert_flash_mode();
    int use_bert_flash = out_16 && c->family == FFWD_FAMILY_BERT &&
                         (out_dtype == CUDA_DTYPE_BF16 || out_dtype == CUDA_DTYPE_F16) &&
                         ((hd == 64 && bert_flash >= 0) || (hd == 32 && bert_flash > 0));
    if (!qkv_src)
        qkv_src = ctx->qkv;
    if (qkv_dtype != CUDA_DTYPE_F32 && (!use_bert_flash || qkv_dtype != out_dtype)) {
        fprintf(stderr, "cuda: reduced-precision qkv is only supported by matching BERT flash\n");
        return -1;
    }
    int direct_kv_mode = bert_direct_kv_mode();
    int use_bert_direct_kv = use_bert_flash && qkv_dtype != CUDA_DTYPE_F32 &&
                             qkv_dtype == out_dtype && direct_kv_mode >= 0;

    int threads = 256;
    size_t exp_count = (size_t)total * q_dim;
    if (!use_bert_direct_kv) {
        if (use_bert_flash && out_dtype == CUDA_DTYPE_BF16 && qkv_dtype == CUDA_DTYPE_BF16) {
            if (!ctx->kexp_bf16)
                return -1;
            attn_expand_kv_kernel<<<(exp_count + threads - 1) / threads, threads, 0, ctx->stream>>>(
                (__nv_bfloat16 *)ctx->kexp_bf16, (__nv_bfloat16 *)ctx->vexp,
                (const __nv_bfloat16 *)qkv_src, total, H, c->n_kv_heads, hd, qkv_dim, k_offset,
                v_offset);
        } else if (use_bert_flash && out_dtype == CUDA_DTYPE_F16 && qkv_dtype == CUDA_DTYPE_F16) {
            if (!ctx->kexp_f16)
                return -1;
            attn_expand_kv_kernel<<<(exp_count + threads - 1) / threads, threads, 0, ctx->stream>>>(
                (__half *)ctx->kexp_f16, (__half *)ctx->vexp, (const __half *)qkv_src, total, H,
                c->n_kv_heads, hd, qkv_dim, k_offset, v_offset);
        } else if (use_tc_kexp_bf16 || (use_bert_flash && out_dtype == CUDA_DTYPE_BF16)) {
            if (!ctx->kexp_bf16)
                return -1;
            attn_expand_kv_kernel<<<(exp_count + threads - 1) / threads, threads, 0, ctx->stream>>>(
                (__nv_bfloat16 *)ctx->kexp_bf16, (__nv_bfloat16 *)ctx->vexp, (const float *)qkv_src,
                total, H, c->n_kv_heads, hd, qkv_dim, k_offset, v_offset);
        } else if (use_bert_flash && out_dtype == CUDA_DTYPE_F16) {
            if (!ctx->kexp_f16)
                return -1;
            attn_expand_kv_kernel<<<(exp_count + threads - 1) / threads, threads, 0, ctx->stream>>>(
                (__half *)ctx->kexp_f16, (__half *)ctx->vexp, (const float *)qkv_src, total, H,
                c->n_kv_heads, hd, qkv_dim, k_offset, v_offset);
        } else if (out_dtype == CUDA_DTYPE_BF16) {
            attn_expand_kv_kernel<<<(exp_count + threads - 1) / threads, threads, 0, ctx->stream>>>(
                ctx->kexp, (__nv_bfloat16 *)ctx->vexp, (const float *)qkv_src, total, H,
                c->n_kv_heads, hd, qkv_dim, k_offset, v_offset);
        } else if (out_dtype == CUDA_DTYPE_F16) {
            attn_expand_kv_kernel<<<(exp_count + threads - 1) / threads, threads, 0, ctx->stream>>>(
                ctx->kexp, (__half *)ctx->vexp, (const float *)qkv_src, total, H, c->n_kv_heads, hd,
                qkv_dim, k_offset, v_offset);
        } else {
            attn_expand_kv_kernel<<<(exp_count + threads - 1) / threads, threads, 0, ctx->stream>>>(
                ctx->kexp, ctx->vexp, (const float *)qkv_src, total, H, c->n_kv_heads, hd, qkv_dim,
                k_offset, v_offset);
        }
        if (launch_check() != 0)
            return -1;
    }

    if (use_bert_flash) {
        int w = tc_kv_warps_or(8);
#define BERT_FLASH_LAUNCH(HDV, W)                                                                    \
    do {                                                                                             \
        dim3 grid((max_L + (W) * 16 - 1) / ((W) * 16), H, batch);                                    \
        dim3 block(32, (W));                                                                         \
        if (out_dtype == CUDA_DTYPE_F16) {                                                           \
            if (qkv_dtype == CUDA_DTYPE_F16 && use_bert_direct_kv)                                   \
                if (bert_flash_pipeline)                                                             \
                    attn_stream_bert_flash_f16_kernel<(HDV), (W), __half, 1, 1>                      \
                        <<<grid, block, 0, ctx->stream>>>(                                           \
                            (__half *)out_16, (const __half *)qkv_src,                               \
                            (const __half *)ctx->kexp_f16, (const __half *)ctx->vexp, ctx->offsets,  \
                            q_dim, qkv_dim, q_offset, k_offset, v_offset, H, c->n_kv_heads, scale);  \
                else                                                                                 \
                    attn_stream_bert_flash_f16_kernel<(HDV), (W), __half, 1, 0>                      \
                        <<<grid, block, 0, ctx->stream>>>(                                           \
                            (__half *)out_16, (const __half *)qkv_src,                               \
                            (const __half *)ctx->kexp_f16, (const __half *)ctx->vexp, ctx->offsets,  \
                            q_dim, qkv_dim, q_offset, k_offset, v_offset, H, c->n_kv_heads, scale);  \
            else if (qkv_dtype == CUDA_DTYPE_F16)                                                    \
                if (bert_flash_pipeline)                                                             \
                    attn_stream_bert_flash_f16_kernel<(HDV), (W), __half, 0, 1>                      \
                        <<<grid, block, 0, ctx->stream>>>(                                           \
                            (__half *)out_16, (const __half *)qkv_src,                               \
                            (const __half *)ctx->kexp_f16, (const __half *)ctx->vexp, ctx->offsets,  \
                            q_dim, qkv_dim, q_offset, k_offset, v_offset, H, c->n_kv_heads, scale);  \
                else                                                                                 \
                    attn_stream_bert_flash_f16_kernel<(HDV), (W), __half, 0, 0>                      \
                        <<<grid, block, 0, ctx->stream>>>(                                           \
                            (__half *)out_16, (const __half *)qkv_src,                               \
                            (const __half *)ctx->kexp_f16, (const __half *)ctx->vexp, ctx->offsets,  \
                            q_dim, qkv_dim, q_offset, k_offset, v_offset, H, c->n_kv_heads, scale);  \
            else if (bert_flash_pipeline)                                                            \
                attn_stream_bert_flash_f16_kernel<(HDV), (W), float, 0, 1>                           \
                    <<<grid, block, 0, ctx->stream>>>(                                               \
                        (__half *)out_16, (const float *)qkv_src, (const __half *)ctx->kexp_f16,     \
                        (const __half *)ctx->vexp, ctx->offsets, q_dim, qkv_dim, q_offset, k_offset, \
                        v_offset, H, c->n_kv_heads, scale);                                          \
            else                                                                                     \
                attn_stream_bert_flash_f16_kernel<(HDV), (W), float, 0, 0>                           \
                    <<<grid, block, 0, ctx->stream>>>(                                               \
                        (__half *)out_16, (const float *)qkv_src, (const __half *)ctx->kexp_f16,     \
                        (const __half *)ctx->vexp, ctx->offsets, q_dim, qkv_dim, q_offset, k_offset, \
                        v_offset, H, c->n_kv_heads, scale);                                          \
        } else {                                                                                     \
            if (qkv_dtype == CUDA_DTYPE_BF16 && use_bert_direct_kv)                                  \
                if (bert_flash_pipeline)                                                             \
                    attn_stream_bert_flash_kernel<(HDV), (W), __nv_bfloat16, 1, 1>                   \
                        <<<grid, block, 0, ctx->stream>>>(                                           \
                            (__nv_bfloat16 *)out_16, (const __nv_bfloat16 *)qkv_src,                 \
                            (const __nv_bfloat16 *)ctx->kexp_bf16, (const __nv_bfloat16 *)ctx->vexp, \
                            ctx->offsets, q_dim, qkv_dim, q_offset, k_offset, v_offset, H,           \
                            c->n_kv_heads, scale);                                                   \
                else                                                                                 \
                    attn_stream_bert_flash_kernel<(HDV), (W), __nv_bfloat16, 1, 0>                   \
                        <<<grid, block, 0, ctx->stream>>>(                                           \
                            (__nv_bfloat16 *)out_16, (const __nv_bfloat16 *)qkv_src,                 \
                            (const __nv_bfloat16 *)ctx->kexp_bf16, (const __nv_bfloat16 *)ctx->vexp, \
                            ctx->offsets, q_dim, qkv_dim, q_offset, k_offset, v_offset, H,           \
                            c->n_kv_heads, scale);                                                   \
            else if (qkv_dtype == CUDA_DTYPE_BF16)                                                   \
                if (bert_flash_pipeline)                                                             \
                    attn_stream_bert_flash_kernel<(HDV), (W), __nv_bfloat16, 0, 1>                   \
                        <<<grid, block, 0, ctx->stream>>>(                                           \
                            (__nv_bfloat16 *)out_16, (const __nv_bfloat16 *)qkv_src,                 \
                            (const __nv_bfloat16 *)ctx->kexp_bf16, (const __nv_bfloat16 *)ctx->vexp, \
                            ctx->offsets, q_dim, qkv_dim, q_offset, k_offset, v_offset, H,           \
                            c->n_kv_heads, scale);                                                   \
                else                                                                                 \
                    attn_stream_bert_flash_kernel<(HDV), (W), __nv_bfloat16, 0, 0>                   \
                        <<<grid, block, 0, ctx->stream>>>(                                           \
                            (__nv_bfloat16 *)out_16, (const __nv_bfloat16 *)qkv_src,                 \
                            (const __nv_bfloat16 *)ctx->kexp_bf16, (const __nv_bfloat16 *)ctx->vexp, \
                            ctx->offsets, q_dim, qkv_dim, q_offset, k_offset, v_offset, H,           \
                            c->n_kv_heads, scale);                                                   \
            else if (bert_flash_pipeline)                                                            \
                attn_stream_bert_flash_kernel<(HDV), (W), float, 0, 1>                               \
                    <<<grid, block, 0, ctx->stream>>>(                                               \
                        (__nv_bfloat16 *)out_16, (const float *)qkv_src,                             \
                        (const __nv_bfloat16 *)ctx->kexp_bf16, (const __nv_bfloat16 *)ctx->vexp,     \
                        ctx->offsets, q_dim, qkv_dim, q_offset, k_offset, v_offset, H,               \
                        c->n_kv_heads, scale);                                                       \
            else                                                                                     \
                attn_stream_bert_flash_kernel<(HDV), (W), float, 0, 0>                               \
                    <<<grid, block, 0, ctx->stream>>>(                                               \
                        (__nv_bfloat16 *)out_16, (const float *)qkv_src,                             \
                        (const __nv_bfloat16 *)ctx->kexp_bf16, (const __nv_bfloat16 *)ctx->vexp,     \
                        ctx->offsets, q_dim, qkv_dim, q_offset, k_offset, v_offset, H,               \
                        c->n_kv_heads, scale);                                                       \
        }                                                                                            \
    } while (0)
        if (hd == 32) {
            if (w == 2)
                BERT_FLASH_LAUNCH(32, 2);
            else if (w == 4)
                BERT_FLASH_LAUNCH(32, 4);
            else
                BERT_FLASH_LAUNCH(32, 8);
        } else {
            if (w == 2)
                BERT_FLASH_LAUNCH(64, 2);
            else if (w == 4)
                BERT_FLASH_LAUNCH(64, 4);
            else
                BERT_FLASH_LAUNCH(64, 8);
        }
#undef BERT_FLASH_LAUNCH
        return launch_check();
    }

    if (use_stream) {
        // stream_mode 0 is the promoted default (use_streaming_attention only
        // returns true for mode 0 on this Qwen BF16 HD=128 case); mode 8 is the
        // explicit `tc3` override. Both run the same kernel.
        if ((stream_mode == 8 || stream_mode == 0) && c->family == FFWD_FAMILY_QWEN3 &&
            out_dtype == CUDA_DTYPE_BF16 && out_16 && hd == 128) {
            // Promoted Qwen BF16 tensor-core flash attention.
            int w = tc_kv_warps_or(8);
#define TC3_LAUNCH(W)                                                                             \
    do {                                                                                          \
        dim3 grid((max_L + (W) * 16 - 1) / ((W) * 16), H, batch);                                 \
        dim3 block(32, (W));                                                                      \
        if (causal)                                                                               \
            attn_stream_tc128_bf16_kv_ldm_kernel<1, (W)><<<grid, block, 0, ctx->stream>>>(        \
                (__nv_bfloat16 *)out_16, ctx->qkv, (const __nv_bfloat16 *)ctx->kexp_bf16,         \
                (const __nv_bfloat16 *)ctx->vexp, ctx->offsets, q_dim, qkv_dim, q_offset, scale); \
        else                                                                                      \
            attn_stream_tc128_bf16_kv_ldm_kernel<0, (W)><<<grid, block, 0, ctx->stream>>>(        \
                (__nv_bfloat16 *)out_16, ctx->qkv, (const __nv_bfloat16 *)ctx->kexp_bf16,         \
                (const __nv_bfloat16 *)ctx->vexp, ctx->offsets, q_dim, qkv_dim, q_offset, scale); \
    } while (0)
            if (w == 2)
                TC3_LAUNCH(2);
            else if (w == 8)
                TC3_LAUNCH(8);
            else
                TC3_LAUNCH(4);
#undef TC3_LAUNCH
            return launch_check();
        }
    }

    if (ctx->attn_G >= 2) {
        int G = ctx->attn_G, L = ctx->attn_L;
        const void *const *K = (const void *const *)ctx->attn_ptrs;
        const void *const *Q = K + (size_t)batch * H;
        void *const *S = (void *const *)(Q + (size_t)batch * H);
        const void *const *V = (const void *const *)S + (size_t)batch * H;
        const void *const *P = V + (size_t)batch * H;
        void *const *O = (void *const *)(P + (size_t)batch * H);
        int use_16 = out_dtype == CUDA_DTYPE_BF16 || out_dtype == CUDA_DTYPE_F16;
        cudaDataType pv = out_dtype == CUDA_DTYPE_BF16
                              ? CUDA_R_16BF
                              : (out_dtype == CUDA_DTYPE_F16 ? CUDA_R_16F : CUDA_R_32F);
        cublasComputeType_t av_ct = use_16 ? CUBLAS_COMPUTE_32F : gemm_compute();
        for (int s = 0; s < batch; s += G) {
            int g = batch - s < G ? batch - s : G;
            int n = g * H;
            size_t off = (size_t)s * H;
            /* scores = scale * Q @ K^T, batched over (seq, head) */
            CUBLAS_CHECK(cublasGemmBatchedEx(ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N, L, L, hd, &alpha,
                                             K + off, CUDA_R_32F, q_dim, Q + off, CUDA_R_32F, qkv_dim,
                                             &beta, S + off, CUDA_R_32F, L, n, gemm_compute(),
                                             CUBLAS_GEMM_DEFAULT));
            if (out_dtype == CUDA_DTYPE_BF16) {
                if (causal)
                    attn_causal_softmax_kernel<<<n * L, 256, 0, ctx->stream>>>(
                        ctx->attn_scores, (__nv_bfloat16 *)ctx->attn_probs, L);
                else
                    attn_softmax_kernel<<<n * L, 256, 0, ctx->stream>>>(
                        ctx->attn_scores, (__nv_bfloat16 *)ctx->attn_probs, L);
            } else if (out_dtype == CUDA_DTYPE_F16) {
                if (causal)
                    attn_causal_softmax_kernel<<<n * L, 256, 0, ctx->stream>>>(
                        ctx->attn_scores, (__half *)ctx->attn_probs, L);
                else
                    attn_softmax_kernel<<<n * L, 256, 0, ctx->stream>>>(ctx->attn_scores,
                                                                        (__half *)ctx->attn_probs, L);
            } else {
                if (causal)
                    attn_causal_softmax_kernel<<<n * L, 256, 0, ctx->stream>>>(ctx->attn_scores,
                                                                               ctx->attn_scores, L);
                else
                    attn_softmax_kernel<<<n * L, 256, 0, ctx->stream>>>(ctx->attn_scores,
                                                                        ctx->attn_scores, L);
            }
            if (launch_check() != 0)
                return -1;
            /* O = P @ V, batched over (seq, head) */
            CUBLAS_CHECK(cublasGemmBatchedEx(ctx->blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, L, L, &one,
                                             V + off, pv, q_dim, P + off, pv, L, &beta, O + off, pv,
                                             q_dim, n, av_ct, CUBLAS_GEMM_DEFAULT));
        }
        return 0;
    }

    for (int b = 0; b < batch; b++) {
        int start = h_offsets[b];
        int L = h_offsets[b + 1] - start;
        const float *Q = ctx->qkv + (size_t)start * qkv_dim + q_offset;
        const float *K = ctx->kexp + (size_t)start * q_dim;
        /* scores[h] = scale * Q[h] @ K[h]^T  (row-major C = A @ B^T) */
        if (gemm_strided_batched(ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N, L, L, hd, &alpha, K, q_dim, hd,
                                 Q, qkv_dim, hd, &beta, ctx->attn_scores, L, (long long)L * L,
                                 H) != 0)
            return -1;
        if (out_dtype == CUDA_DTYPE_BF16 || out_dtype == CUDA_DTYPE_F16) {
            cudaDataType pv = out_dtype == CUDA_DTYPE_BF16 ? CUDA_R_16BF : CUDA_R_16F;
            void *P = (uint16_t *)ctx->attn_probs;
            const void *V = (const uint16_t *)ctx->vexp + (size_t)start * q_dim;
            void *O = (uint16_t *)out_16 + (size_t)start * q_dim;
            if (causal) {
                if (out_dtype == CUDA_DTYPE_BF16)
                    attn_causal_softmax_kernel<<<H * L, 256, 0, ctx->stream>>>(ctx->attn_scores,
                                                                               (__nv_bfloat16 *)P, L);
                else
                    attn_causal_softmax_kernel<<<H * L, 256, 0, ctx->stream>>>(ctx->attn_scores,
                                                                               (__half *)P, L);
            } else {
                if (out_dtype == CUDA_DTYPE_BF16)
                    attn_softmax_kernel<<<H * L, 256, 0, ctx->stream>>>(ctx->attn_scores,
                                                                        (__nv_bfloat16 *)P, L);
                else
                    attn_softmax_kernel<<<H * L, 256, 0, ctx->stream>>>(ctx->attn_scores, (__half *)P,
                                                                        L);
            }
            if (launch_check() != 0)
                return -1;
            /* O[h] = P[h] @ V[h]  (row-major C = A @ B) */
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(ctx->blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, L, L,
                                                    &one, V, pv, q_dim, hd, P, pv, L,
                                                    (long long)L * L, &beta, O, pv, q_dim, hd, H,
                                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        } else {
            const float *V = ctx->vexp + (size_t)start * q_dim;
            float *O = ctx->attn_out + (size_t)start * q_dim;
            if (causal)
                attn_causal_softmax_kernel<<<H * L, 256, 0, ctx->stream>>>(ctx->attn_scores,
                                                                           ctx->attn_scores, L);
            else
                attn_softmax_kernel<<<H * L, 256, 0, ctx->stream>>>(ctx->attn_scores,
                                                                    ctx->attn_scores, L);
            if (launch_check() != 0)
                return -1;
            /* O[h] = P[h] @ V[h]  (row-major C = A @ B) */
            if (gemm_strided_batched(ctx->blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, L, L, &one, V, q_dim,
                                     hd, ctx->attn_scores, L, (long long)L * L, &beta, O, q_dim, hd,
                                     H) != 0)
                return -1;
        }
    }
    return 0;
}

enum { BERT_BF16_ACT_MIN_TOKENS = 1024 };
enum { BERT_RESIDUAL16_MIN_TOKENS = 16384 };
enum { BERT_FFN16_MIN_TOKENS = 1024 };
enum { BERT_QKV16_MIN_TOKENS = 1024 };

typedef struct {
    int compute_16bit;
    int bf16_flash;
    int f16_flash;
    int flash;
    int bf16_act;
    int f16_act;
    cuda_dtype_t act_dtype;
    int bf16_resid;
    int f16_resid;
    int use_resid16;
    int ffn16;
    int delta16;
    int cublaslt_avail;
    int cublaslt;
    int qkv16;
    int direct_kv;
} bert_cuda_plan_t;

static int bert_flash_mode(void);
static int bert_direct_kv_mode(void);
static int bert_bf16_act_mode(void);
static int bert_residual16_mode(void);
static int bert_ffn16_mode(void);
static int bert_delta16_mode(void);
static int bert_cublaslt_mode(void);
static int bert_qkv16_mode(void);
static int bert_flash_pipeline_mode(void);

static void bert_cuda_plan(ffwd_cuda_ctx_t *ctx, int total, bert_cuda_plan_t *p) {
    memset(p, 0, sizeof(*p));
    p->act_dtype = CUDA_DTYPE_F32;
    if (!ctx || ctx->config.family != FFWD_FAMILY_BERT || ctx->config.n_layers <= 0)
        return;

    const ffwd_config_t *c = &ctx->config;
    cublasComputeType_t gc = gemm_compute();
    p->compute_16bit = (gc == CUBLAS_COMPUTE_32F_FAST_16BF || gc == CUBLAS_COMPUTE_32F_FAST_16F);

    int bert_flash = bert_flash_mode();
    p->bf16_flash = p->compute_16bit && ctx->layers[0].qkv.bf16 &&
                    ((c->head_dim == 64 && bert_flash >= 0) || (c->head_dim == 32 && bert_flash > 0));
    p->f16_flash = p->compute_16bit && ctx->layers[0].qkv.f16 &&
                   ((c->head_dim == 64 && bert_flash >= 0) || (c->head_dim == 32 && bert_flash > 0));
    p->flash = p->bf16_flash || p->f16_flash;

    int act_mode = bert_bf16_act_mode();
    p->bf16_act = p->bf16_flash || (p->compute_16bit && ctx->layers[0].qkv.bf16 && act_mode >= 0 &&
                                    (act_mode == 1 || total >= BERT_BF16_ACT_MIN_TOKENS));
    p->f16_act = p->f16_flash;
    p->act_dtype = p->f16_act ? CUDA_DTYPE_F16 : (p->bf16_act ? CUDA_DTYPE_BF16 : CUDA_DTYPE_F32);

    int resid_mode = bert_residual16_mode();
    p->bf16_resid = p->bf16_act && p->flash && resid_mode >= 0 &&
                    (resid_mode > 0 || total >= BERT_RESIDUAL16_MIN_TOKENS);
    p->f16_resid = p->f16_act && p->flash && resid_mode >= 0;
    p->use_resid16 = p->bf16_resid || p->f16_resid;

    int ffn16_mode = bert_ffn16_mode();
    p->ffn16 = p->act_dtype != CUDA_DTYPE_F32 && ffn16_mode >= 0 &&
               (ffn16_mode > 0 || total >= BERT_FFN16_MIN_TOKENS);
    p->delta16 = p->use_resid16 && bert_delta16_mode() >= 0;

    p->cublaslt_avail =
        bert_cublaslt_mode() >= 0 && p->act_dtype != CUDA_DTYPE_F32 && ctx->lt != NULL;
    p->cublaslt = p->cublaslt_avail && c->ffn_act == FFWD_ACT_GELU_ERF;

    int qkv16_mode = bert_qkv16_mode();
    p->qkv16 = p->flash && p->cublaslt_avail && qkv16_mode >= 0 &&
               (qkv16_mode > 0 || total >= BERT_QKV16_MIN_TOKENS);
    p->direct_kv = p->flash && p->qkv16 && bert_direct_kv_mode() >= 0;
}

static int ensure_buffers(ffwd_cuda_ctx_t *ctx, int total, int batch, int max_seq) {
    const ffwd_config_t *c = &ctx->config;
    bert_cuda_plan_t bp;
    bert_cuda_plan(ctx, total, &bp);

    int is_bert = c->family == FFWD_FAMILY_BERT;
    int need_x = 1;
    int need_x_norm = !is_bert;
    int need_qkv = !is_bert || !bp.qkv16;
    int need_qkv_16 = is_bert && bp.qkv16;
    int need_attn_out = !is_bert || !bp.flash;
    int need_ffn_gate_up = 1;
    int need_ffn_gate_up_16 = 0;
    int need_resid_delta_16 = is_bert && bp.delta16;
    int need_x_bf16 = is_bert && bp.bf16_resid;
    int need_x_f16 = is_bert && bp.f16_resid;
    int need_kexp = 1;
    int need_kexp_bf16 = ctx->weights_bf16;
    int need_kexp_f16 = ctx->weights_f16;
    int need_vexp = 1;
    int vexp_elem_bytes = (int)sizeof(float);
    int need_act_bf16 = ctx->weights_bf16;
    int need_act_f16 = ctx->weights_f16;
    int act_cols = 2 * c->intermediate_size;
    int ffn_gate_up_cols = 2 * c->intermediate_size;

    if (is_bert) {
        need_x_norm = 0;
        need_ffn_gate_up = !bp.cublaslt && (bp.act_dtype == CUDA_DTYPE_F32 || !bp.ffn16);
        // The cuBLASLt fused gate-up also needs this dedicated 16-bit intermediate
        // so its output does not alias its norm_act input (see the FFN forward).
        need_ffn_gate_up_16 = bp.act_dtype != CUDA_DTYPE_F32 && (bp.ffn16 || bp.cublaslt);
        ffn_gate_up_cols = c->intermediate_size;
        need_kexp = !bp.flash || !bp.direct_kv;
        need_kexp_bf16 = bp.flash && !bp.direct_kv && bp.act_dtype == CUDA_DTYPE_BF16;
        need_kexp_f16 = bp.flash && !bp.direct_kv && bp.act_dtype == CUDA_DTYPE_F16;
        need_vexp = !bp.flash || !bp.direct_kv;
        vexp_elem_bytes =
            bp.flash && bp.act_dtype != CUDA_DTYPE_F32 ? (int)sizeof(uint16_t) : (int)sizeof(float);
        need_act_bf16 = bp.act_dtype == CUDA_DTYPE_BF16;
        need_act_f16 = bp.act_dtype == CUDA_DTYPE_F16;
        act_cols = c->intermediate_size;
        if (act_cols < c->hidden_size)
            act_cols = c->hidden_size;
        if (act_cols < c->q_dim)
            act_cols = c->q_dim;
    }

    int scratch_key = 1;
    scratch_key |= need_x_norm << 1;
    scratch_key |= need_qkv << 2;
    scratch_key |= need_qkv_16 << 3;
    scratch_key |= need_attn_out << 4;
    scratch_key |= need_ffn_gate_up << 5;
    scratch_key |= need_ffn_gate_up_16 << 6;
    scratch_key |= need_resid_delta_16 << 7;
    scratch_key |= need_x_bf16 << 8;
    scratch_key |= need_x_f16 << 9;
    scratch_key |= need_kexp << 10;
    scratch_key |= need_kexp_bf16 << 11;
    scratch_key |= need_kexp_f16 << 12;
    scratch_key |= need_vexp << 13;
    scratch_key |= (vexp_elem_bytes == (int)sizeof(uint16_t)) << 14;
    scratch_key |= need_act_bf16 << 15;
    scratch_key |= need_act_f16 << 16;
    scratch_key |= is_bert << 17;
    scratch_key |= (bp.flash ? 1 : 0) << 18;

    if (total > ctx->seq_cap || scratch_key != ctx->scratch_key) {
        cudaFree(ctx->x);
        cudaFree(ctx->x_norm);
        cudaFree(ctx->qkv);
        cudaFree(ctx->qkv_16);
        cudaFree(ctx->attn_out);
        cudaFree(ctx->ffn_gate_up);
        cudaFree(ctx->ffn_gate_up_16);
        cudaFree(ctx->resid_delta_16);
        cudaFree(ctx->x_bf16);
        cudaFree(ctx->x_f16);
        cudaFree(ctx->token_ids);
        cudaFree(ctx->positions);
        cudaFree(ctx->kexp);
        cudaFree(ctx->kexp_bf16);
        cudaFree(ctx->kexp_f16);
        cudaFree(ctx->vexp);
        cudaFree(ctx->act_bf16);
        cudaFree(ctx->act_f16);
        cudaFree(ctx->rope_cos);
        cudaFree(ctx->rope_sin);
        cudaFree(ctx->attn_scores);
        cudaFree(ctx->attn_probs);
        ctx->x = ctx->x_norm = ctx->qkv = NULL;
        ctx->qkv_16 = NULL;
        ctx->attn_out = ctx->ffn_gate_up = NULL;
        ctx->ffn_gate_up_16 = NULL;
        ctx->resid_delta_16 = NULL;
        ctx->x_bf16 = NULL;
        ctx->x_f16 = NULL;
        ctx->token_ids = ctx->positions = NULL;
        ctx->kexp = ctx->vexp = NULL;
        ctx->kexp_bf16 = NULL;
        ctx->kexp_f16 = NULL;
        ctx->act_bf16 = NULL;
        ctx->act_f16 = NULL;
        ctx->rope_cos = NULL;
        ctx->rope_sin = NULL;
        ctx->attn_scores = NULL;
        ctx->attn_probs = NULL;
        ctx->attn_scores_elems = 0;
        ctx->attn_probs_elems = 0;
        ctx->max_seq_cap = 0;
        ctx->scratch_key = 0;
        if (need_x)
            CUDA_CHECK(cudaMalloc((void **)&ctx->x, (size_t)total * c->hidden_size * sizeof(float)));
        if (need_x_norm)
            CUDA_CHECK(
                cudaMalloc((void **)&ctx->x_norm, (size_t)total * c->hidden_size * sizeof(float)));
        if (need_qkv)
            CUDA_CHECK(cudaMalloc((void **)&ctx->qkv,
                                  (size_t)total * (c->q_dim + 2 * c->kv_dim) * sizeof(float)));
        if (need_qkv_16)
            CUDA_CHECK(cudaMalloc(&ctx->qkv_16,
                                  (size_t)total * (c->q_dim + 2 * c->kv_dim) * sizeof(uint16_t)));
        if (need_attn_out)
            CUDA_CHECK(cudaMalloc((void **)&ctx->attn_out, (size_t)total * c->q_dim * sizeof(float)));
        if (need_ffn_gate_up)
            CUDA_CHECK(cudaMalloc((void **)&ctx->ffn_gate_up,
                                  (size_t)total * ffn_gate_up_cols * sizeof(float)));
        if (need_x_bf16)
            CUDA_CHECK(
                cudaMalloc(&ctx->x_bf16, (size_t)total * c->hidden_size * sizeof(__nv_bfloat16)));
        if (need_x_f16)
            CUDA_CHECK(cudaMalloc(&ctx->x_f16, (size_t)total * c->hidden_size * sizeof(__half)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->token_ids, (size_t)total * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->positions, (size_t)total * sizeof(int)));
        if (need_kexp)
            CUDA_CHECK(cudaMalloc((void **)&ctx->kexp, (size_t)total * c->q_dim * sizeof(float)));
        if (need_kexp_bf16)
            CUDA_CHECK(cudaMalloc(&ctx->kexp_bf16, (size_t)total * c->q_dim * sizeof(__nv_bfloat16)));
        if (need_kexp_f16)
            CUDA_CHECK(cudaMalloc(&ctx->kexp_f16, (size_t)total * c->q_dim * sizeof(__half)));
        if (need_vexp)
            CUDA_CHECK(cudaMalloc((void **)&ctx->vexp, (size_t)total * c->q_dim * vexp_elem_bytes));
        if (need_act_bf16)
            CUDA_CHECK(cudaMalloc(&ctx->act_bf16, (size_t)total * act_cols * sizeof(__nv_bfloat16)));
        if (need_act_f16)
            CUDA_CHECK(cudaMalloc(&ctx->act_f16, (size_t)total * act_cols * sizeof(__half)));
        if (need_ffn_gate_up_16)
            CUDA_CHECK(cudaMalloc(&ctx->ffn_gate_up_16,
                                  (size_t)total * c->intermediate_size * sizeof(uint16_t)));
        if (need_resid_delta_16)
            CUDA_CHECK(
                cudaMalloc(&ctx->resid_delta_16, (size_t)total * c->hidden_size * sizeof(uint16_t)));
        ctx->seq_cap = total;
        ctx->scratch_key = scratch_key;
    }
    if (batch + 1 > ctx->batch_cap) {
        cudaFree(ctx->offsets);
        free(ctx->offsets_host);
        CUDA_CHECK(cudaMalloc((void **)&ctx->offsets, (size_t)(batch + 1) * sizeof(int)));
        ctx->offsets_host = (int *)malloc((size_t)(batch + 1) * sizeof(int));
        if (!ctx->offsets_host)
            return -1;
        ctx->batch_cap = batch + 1;
    }
    if (max_seq > ctx->max_seq_cap) {
        cudaFree(ctx->rope_cos);
        cudaFree(ctx->rope_sin);
        ctx->rope_cos = NULL;
        ctx->rope_sin = NULL;
        long long score_elems = (long long)c->n_heads * max_seq * max_seq;
        int need_materialized_scores = !is_bert || !bp.flash;
        if (need_materialized_scores && score_elems > ctx->attn_scores_elems) {
            cudaFree(ctx->attn_scores);
            ctx->attn_scores = NULL;
            ctx->attn_scores_elems = 0;
            CUDA_CHECK(cudaMalloc((void **)&ctx->attn_scores, (size_t)score_elems * sizeof(float)));
            ctx->attn_scores_elems = score_elems;
        }
        if (need_materialized_scores && (ctx->weights_bf16 || ctx->weights_f16) &&
            score_elems > ctx->attn_probs_elems) {
            cudaFree(ctx->attn_probs);
            ctx->attn_probs = NULL;
            ctx->attn_probs_elems = 0;
            CUDA_CHECK(cudaMalloc(&ctx->attn_probs, (size_t)score_elems * sizeof(uint16_t)));
            ctx->attn_probs_elems = score_elems;
        }
        if (is_bert) {
            ctx->max_seq_cap = max_seq;
            return 0;
        }
        float *hcos = (float *)malloc((size_t)max_seq * c->head_dim * sizeof(float));
        float *hsin = (float *)malloc((size_t)max_seq * c->head_dim * sizeof(float));
        if (!hcos || !hsin) {
            free(hcos);
            free(hsin);
            return -1;
        }
        int half = c->head_dim / 2;
        for (int pos = 0; pos < max_seq; pos++) {
            for (int d = 0; d < half; d++) {
                float inv = 1.0f / powf(c->rope_theta, (float)(2 * d) / (float)c->head_dim);
                float angle = (float)pos * inv;
                float cv = cosf(angle);
                float sv = sinf(angle);
                hcos[(size_t)pos * c->head_dim + d] = cv;
                hcos[(size_t)pos * c->head_dim + half + d] = cv;
                hsin[(size_t)pos * c->head_dim + d] = sv;
                hsin[(size_t)pos * c->head_dim + half + d] = sv;
            }
        }
        CUDA_CHECK(
            cudaMalloc((void **)&ctx->rope_cos, (size_t)max_seq * c->head_dim * sizeof(float)));
        CUDA_CHECK(
            cudaMalloc((void **)&ctx->rope_sin, (size_t)max_seq * c->head_dim * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(ctx->rope_cos, hcos, (size_t)max_seq * c->head_dim * sizeof(float),
                                   cudaMemcpyHostToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpyAsync(ctx->rope_sin, hsin, (size_t)max_seq * c->head_dim * sizeof(float),
                                   cudaMemcpyHostToDevice, ctx->stream));
        free(hcos);
        free(hsin);
        ctx->max_seq_cap = max_seq;
    }
    return 0;
}

static int ensure_pooled_rows(ffwd_cuda_ctx_t *ctx, int rows) {
    if (rows <= ctx->pooled_rows_cap)
        return 0;
    const ffwd_config_t *c = &ctx->config;
    cudaFree(ctx->pooled_out);
    ctx->pooled_out = NULL;
    CUDA_CHECK(cudaMalloc((void **)&ctx->pooled_out, (size_t)rows * c->hidden_size * sizeof(float)));
    ctx->pooled_rows_cap = rows;
    return 0;
}

static int ensure_span_buffers(ffwd_cuda_ctx_t *ctx, int n_spans) {
    if (n_spans <= ctx->span_cap)
        return 0;
    cudaFree(ctx->span_starts);
    cudaFree(ctx->span_lens);
    ctx->span_starts = NULL;
    ctx->span_lens = NULL;
    CUDA_CHECK(cudaMalloc((void **)&ctx->span_starts, (size_t)n_spans * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&ctx->span_lens, (size_t)n_spans * sizeof(int)));
    ctx->span_cap = n_spans;
    return 0;
}

static int load_layer(cuda_layer_t *dst,
                      const ffwd_layer_t *src,
                      const ffwd_config_t *c,
                      cuda_dtype_t weight_dtype) {
    return load_qkv_matrix(&dst->qkv, src, c, weight_dtype) ||
           load_matrix(&dst->wo, &src->wo, c->hidden_size, c->q_dim, weight_dtype) ||
           (c->qkv_bias && load_qkv_bias(&dst->qkv_bias, src, c)) ||
           (c->qk_norm && load_vector(&dst->q_norm, src->q_norm, c->head_dim)) ||
           (c->qk_norm && load_vector(&dst->k_norm, src->k_norm, c->head_dim)) ||
           load_vector(&dst->input_norm, src->input_norm, c->hidden_size) ||
           load_vector(&dst->post_attn_norm, src->post_attn_norm, c->hidden_size) ||
           load_gate_up_matrix(&dst->gate_up_proj, src, c, weight_dtype) ||
           load_matrix(&dst->down_proj, &src->down_proj, c->hidden_size, c->intermediate_size,
                       weight_dtype);
}

/* BERT layer: attention reuses the fused qkv matrix and qkv_bias (q_dim ==
 * kv_dim == hidden); the two block LayerNorm weights load into input_norm
 * (post-attention) and post_attn_norm (post-FFN); gate_up_proj holds the single
 * up_proj. The added fields carry the biases the Qwen slots lack. */
static int load_bert_layer(cuda_layer_t *dst,
                           const ffwd_layer_t *src,
                           const ffwd_config_t *c,
                           cuda_dtype_t weight_dtype) {
    return load_qkv_matrix(&dst->qkv, src, c, weight_dtype) ||
           load_matrix(&dst->wo, &src->wo, c->hidden_size, c->q_dim, weight_dtype) ||
           load_qkv_bias(&dst->qkv_bias, src, c) ||
           load_vector(&dst->o_bias, src->o_bias, c->hidden_size) ||
           load_vector(&dst->input_norm, src->input_norm, c->hidden_size) ||
           load_vector(&dst->attn_ln_bias, src->attn_ln_bias, c->hidden_size) ||
           load_vector(&dst->post_attn_norm, src->post_attn_norm, c->hidden_size) ||
           load_vector(&dst->ffn_ln_bias, src->ffn_ln_bias, c->hidden_size) ||
           load_matrix(&dst->gate_up_proj, &src->up_proj, c->intermediate_size, c->hidden_size,
                       weight_dtype) ||
           load_vector(&dst->ffn_inter_bias, src->ffn_inter_bias, c->intermediate_size) ||
           load_matrix(&dst->down_proj, &src->down_proj, c->hidden_size, c->intermediate_size,
                       weight_dtype) ||
           load_vector(&dst->ffn_out_bias, src->ffn_out_bias, c->hidden_size);
}

static void free_layer(cuda_layer_t *l) {
    cuda_matrix_free(&l->qkv);
    cuda_matrix_free(&l->wo);
    cudaFree(l->qkv_bias);
    cudaFree(l->q_norm);
    cudaFree(l->k_norm);
    cudaFree(l->input_norm);
    cudaFree(l->post_attn_norm);
    cuda_matrix_free(&l->gate_up_proj);
    cuda_matrix_free(&l->down_proj);
    cudaFree(l->o_bias);
    cudaFree(l->attn_ln_bias);
    cudaFree(l->ffn_inter_bias);
    cudaFree(l->ffn_out_bias);
    cudaFree(l->ffn_ln_bias);
}

/* Build a device context from an already-loaded CPU model. own_cpu records
 * whether ffwd_cuda_free should release that CPU model: ffwd_cuda_load owns
 * the model it loaded; the late path passes a model owned by its CPU late
 * model and sets own_cpu = 0. */
static ffwd_cuda_ctx_t *cuda_ctx_from_model(ffwd_model_t *cpu, int own_cpu) {
    if (!cpu)
        return NULL;

    ffwd_cuda_ctx_t *ctx = (ffwd_cuda_ctx_t *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        if (own_cpu)
            ffwd_model_free(cpu);
        return NULL;
    }
    ctx->cpu = cpu;
    ctx->own_cpu = own_cpu;
    ctx->config = cpu->config;
    ctx->weight_dtype =
        g_weight_dtype_override >= 0
            ? (cuda_dtype_t)g_weight_dtype_override
            : (cpu->weights.embed_tokens.dtype == DTYPE_BF16 ? CUDA_DTYPE_BF16 : CUDA_DTYPE_F32);
    ctx->weights_bf16 = ctx->weight_dtype == CUDA_DTYPE_BF16;
    ctx->weights_f16 = ctx->weight_dtype == CUDA_DTYPE_F16;

    /* Without an explicit --gpu-weight-dtype, store weights in the model
     * file's dtype: BF16 snapshots load as BF16 (bit-exact pass-through),
     * F32 snapshots keep the exact F32 default. */
    if (g_weight_dtype_override < 0 && ctx->weights_bf16)
        fprintf(stderr, "cuda: BF16 model file; storing weights as BF16 "
                        "(use --gpu-weight-dtype f32 to override)\n");

    if (cudaStreamCreate(&ctx->stream) != cudaSuccess ||
        cublasCreate(&ctx->blas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(ctx->blas, ctx->stream) != CUBLAS_STATUS_SUCCESS) {
        ffwd_cuda_free(ctx);
        return NULL;
    }
    // cuBLASLt handle + workspace + a small 16-bit bias scratch for the fused
    // bias/GeLU epilogue GEMMs (the bias is cast to the output dtype per call).
    ctx->lt_ws_bytes = (size_t)32 << 20;
    int lt_bias_elems = ctx->config.intermediate_size;
    int qkv_bias_elems = ctx->config.q_dim + 2 * ctx->config.kv_dim;
    if (qkv_bias_elems > lt_bias_elems)
        lt_bias_elems = qkv_bias_elems;
    if (cublasLtCreate(&ctx->lt) != CUBLAS_STATUS_SUCCESS ||
        cudaMalloc(&ctx->lt_workspace, ctx->lt_ws_bytes) != cudaSuccess ||
        cudaMalloc(&ctx->lt_bias16, (size_t)lt_bias_elems * sizeof(uint16_t)) != cudaSuccess) {
        ffwd_cuda_free(ctx);
        return NULL;
    }

    const ffwd_config_t *c = &ctx->config;
    ctx->layers = (cuda_layer_t *)calloc((size_t)c->n_layers, sizeof(*ctx->layers));
    if (!ctx->layers) {
        ffwd_cuda_free(ctx);
        return NULL;
    }
    if (load_matrix(&ctx->embed_tokens, &cpu->weights.embed_tokens, c->vocab_size, c->hidden_size,
                    ctx->weight_dtype) != 0) {
        ffwd_cuda_free(ctx);
        return NULL;
    }
    if (c->family == FFWD_FAMILY_BERT) {
        /* BERT has no final norm; it adds learned position + token-type[0]
         * embeddings and an embedding LayerNorm (only row 0 of token-type is
         * ever used, so load just that row). */
        const ffwd_weights_t *w = &cpu->weights;
        if (load_vector(&ctx->position_embeddings, w->position_embeddings,
                        c->max_position_embeddings * c->hidden_size) != 0 ||
            load_vector(&ctx->token_type_embedding, w->token_type_embeddings, c->hidden_size) != 0 ||
            load_vector(&ctx->ffwd_ln_w, w->ffwd_ln_w, c->hidden_size) != 0 ||
            load_vector(&ctx->ffwd_ln_b, w->ffwd_ln_b, c->hidden_size) != 0) {
            ffwd_cuda_free(ctx);
            return NULL;
        }
    } else if (load_vector(&ctx->norm, cpu->weights.norm, c->hidden_size) != 0) {
        ffwd_cuda_free(ctx);
        return NULL;
    }
    for (int i = 0; i < c->n_layers; i++) {
        int rc = c->family == FFWD_FAMILY_BERT
                     ? load_bert_layer(&ctx->layers[i], &cpu->weights.layers[i], c, ctx->weight_dtype)
                     : load_layer(&ctx->layers[i], &cpu->weights.layers[i], c, ctx->weight_dtype);
        if (rc != 0) {
            fprintf(stderr, "cuda: failed to load layer %d\n", i);
            ffwd_cuda_free(ctx);
            return NULL;
        }
    }
    cudaStreamSynchronize(ctx->stream);
    return ctx;
}

ffwd_cuda_ctx_t *ffwd_cuda_load(const char *model_dir) {
    return cuda_ctx_from_model(ffwd_model_load(model_dir), 1);
}

void ffwd_cuda_free(ffwd_cuda_ctx_t *ctx) {
    if (!ctx)
        return;
    cuda_matrix_free(&ctx->embed_tokens);
    if (ctx->layers) {
        for (int i = 0; i < ctx->config.n_layers; i++)
            free_layer(&ctx->layers[i]);
        free(ctx->layers);
    }
    cudaFree(ctx->norm);
    cudaFree(ctx->position_embeddings);
    cudaFree(ctx->token_type_embedding);
    cudaFree(ctx->ffwd_ln_w);
    cudaFree(ctx->ffwd_ln_b);
    cudaFree(ctx->x);
    cudaFree(ctx->x_norm);
    cudaFree(ctx->qkv);
    cudaFree(ctx->qkv_16);
    cudaFree(ctx->attn_out);
    cudaFree(ctx->ffn_gate_up);
    cudaFree(ctx->ffn_gate_up_16);
    cudaFree(ctx->resid_delta_16);
    cudaFree(ctx->x_bf16);
    cudaFree(ctx->x_f16);
    cudaFree(ctx->pooled_out);
    cudaFree(ctx->rope_cos);
    cudaFree(ctx->rope_sin);
    cudaFree(ctx->token_ids);
    cudaFree(ctx->offsets);
    free(ctx->offsets_host);
    cudaFree(ctx->positions);
    cudaFree(ctx->kexp);
    cudaFree(ctx->kexp_bf16);
    cudaFree(ctx->kexp_f16);
    cudaFree(ctx->vexp);
    cudaFree(ctx->act_bf16);
    cudaFree(ctx->act_f16);
    cudaFree(ctx->weight_f32);
    cudaFree(ctx->attn_scores);
    cudaFree(ctx->attn_probs);
    cudaFree((void *)ctx->attn_ptrs);
    free((void *)ctx->attn_ptrs_host);
    cudaFree(ctx->span_starts);
    cudaFree(ctx->span_lens);
    cudaFree(ctx->lt_workspace);
    cudaFree(ctx->lt_bias16);
    linear_lt_plan_free(ctx);
    if (ctx->lt)
        cublasLtDestroy(ctx->lt);
    if (ctx->blas)
        cublasDestroy(ctx->blas);
    if (ctx->stream)
        cudaStreamDestroy(ctx->stream);
    if (ctx->own_cpu)
        ffwd_model_free(ctx->cpu);
    free(ctx);
}

const ffwd_config_t *ffwd_cuda_config(const ffwd_cuda_ctx_t *ctx) {
    return ctx ? &ctx->config : NULL;
}

// Pack inputs into the device scratch every forward shares: token ids, the
// per-sequence positions 0..L-1, and the batch offsets. Grows scratch via
// ensure_buffers. Returns the packed token count in *out_total, or -1 on error.
static int cuda_upload_packed_inputs(ffwd_cuda_ctx_t *ctx,
                                     const ffwd_input_t *inputs,
                                     int batch,
                                     int *out_total) {
    int *h_offsets = (int *)malloc((size_t)(batch + 1) * sizeof(int));
    if (!h_offsets)
        return -1;
    int total = 0, max_seq = 0;
    for (int b = 0; b < batch; b++) {
        if (!inputs[b].ids || inputs[b].n_tokens <= 0) {
            free(h_offsets);
            return -1;
        }
        h_offsets[b] = total;
        total += inputs[b].n_tokens;
        if (inputs[b].n_tokens > max_seq)
            max_seq = inputs[b].n_tokens;
    }
    if (ctx->config.family == FFWD_FAMILY_BERT &&
        max_seq > ctx->config.max_position_embeddings - ctx->config.position_id_offset) {
        free(h_offsets);
        return -1;
    }
    h_offsets[batch] = total;
    int *h_ids = (int *)malloc((size_t)total * sizeof(int));
    int *h_pos = (int *)malloc((size_t)total * sizeof(int));
    if (!h_ids || !h_pos) {
        free(h_offsets);
        free(h_ids);
        free(h_pos);
        return -1;
    }
    for (int b = 0; b < batch; b++) {
        int off = h_offsets[b];
        memcpy(h_ids + off, inputs[b].ids, (size_t)inputs[b].n_tokens * sizeof(int));
        for (int i = 0; i < inputs[b].n_tokens; i++)
            h_pos[off + i] = ctx->config.position_id_offset + i;
    }
    if (ensure_buffers(ctx, total, batch, max_seq) != 0) {
        free(h_offsets);
        free(h_ids);
        free(h_pos);
        return -1;
    }
    memcpy(ctx->offsets_host, h_offsets, (size_t)(batch + 1) * sizeof(int));
    cudaError_t ce = cudaMemcpyAsync(ctx->token_ids, h_ids, (size_t)total * sizeof(int),
                                     cudaMemcpyHostToDevice, ctx->stream);
    if (ce == cudaSuccess)
        ce = cudaMemcpyAsync(ctx->positions, h_pos, (size_t)total * sizeof(int),
                             cudaMemcpyHostToDevice, ctx->stream);
    if (ce == cudaSuccess)
        ce = cudaMemcpyAsync(ctx->offsets, h_offsets, (size_t)(batch + 1) * sizeof(int),
                             cudaMemcpyHostToDevice, ctx->stream);
    free(h_offsets);
    free(h_ids);
    free(h_pos);
    if (ce != cudaSuccess) {
        fprintf(stderr, "cuda input copy failed: %s\n", cudaGetErrorString(ce));
        return -1;
    }
    *out_total = total;
    return 0;
}

// BERT-family forward (post-norm encoder), mirroring the CPU forward_packed_bert.
// Embeddings: token + learned position + token-type[0], then the embedding
// LayerNorm. Each layer: QKV(+bias) from x directly -> bidirectional attention
// -> output dense(+bias) -> residual -> LayerNorm -> dense(+bias) -> GeLU ->
// dense(+bias) -> residual -> LayerNorm. No RoPE, no qk-norm, no final norm.
// bf16-activation BERT path: the norm/GELU producers emit bf16 GEMM inputs (no
// per-GEMM F32->bf16 cast) on the bf16w-bf16 path. Bit-identical to the cast
// path, +11-13% at batch 32, but a fixed per-norm overhead makes it lose on tiny
// inputs (single short requests). So it is auto-enabled only above a measured
// token-count crossover (~512 tokens; 1024 leaves margin). FFWD_CUDA_BERT_BF16_ACT
// overrides: 1/on forces it on at any size, 0/off forces it off (for A/B).
// BF16 residual stream goes further: in the BF16 flash path it stores the
// persistent residual stream itself as BF16, while doing residual-add and
// LayerNorm reductions in F32. It is auto-enabled only for large packed BERT
// batches where L4 validation showed a stable long-sequence win; the env override
// remains for A/B.
// Returns 1 (force on), -1 (force off), or 0 (auto: gate on token count).
static int bert_bf16_act_mode(void) {
    static int init = 0, mode = 0;
    if (!init) {
        const char *e = getenv("FFWD_CUDA_BERT_BF16_ACT");
        if (e && (!strcmp(e, "1") || !strcmp(e, "on")))
            mode = 1;
        else if (e && (!strcmp(e, "0") || !strcmp(e, "off")))
            mode = -1;
        init = 1;
    }
    return mode;
}

// BERT residual-stream storage. 0/default gates on total tokens; 1/on/bf16
// forces it on; 0/off forces it off. The arithmetic stays mixed precision:
// persistent storage is BF16, residual-add and LayerNorm reductions are F32.
static int bert_residual16_mode(void) {
    static int init = 0, mode = 0;
    if (!init) {
        const char *e = getenv("FFWD_CUDA_BERT_RESIDUAL16");
        if (e && (!strcmp(e, "1") || !strcmp(e, "on") || !strcmp(e, "bf16")))
            mode = 1;
        else if (e && (!strcmp(e, "0") || !strcmp(e, "off")))
            mode = -1;
        init = 1;
    }
    return mode;
}

// BERT FFN intermediate storage. 0/default gates on total tokens; 1/on forces
// the gate-up GEMM to write the intermediate as the active 16-bit dtype; 0/off
// keeps the previous F32 intermediate for A/B.
static int bert_ffn16_mode(void) {
    static int init = 0, mode = 0;
    if (!init) {
        const char *e = getenv("FFWD_CUDA_BERT_FFN16");
        if (e && (!strcmp(e, "1") || !strcmp(e, "on") || !strcmp(e, "force")))
            mode = 1;
        else if (e && (!strcmp(e, "0") || !strcmp(e, "off")))
            mode = -1;
        init = 1;
    }
    return mode;
}

// BERT reduced-precision residual delta. Rides the residual16 gate (on whenever
// the 16-bit residual stream is active); 0/off forces the F32 projection delta
// back for A/B, 1/on is explicit-on. It writes the wo/down projection output as
// the active 16-bit dtype (read back by the *_delta16 LayerNorm kernels) instead
// of an F32 delta.
static int bert_delta16_mode(void) {
    static int init = 0, mode = 0;
    if (!init) {
        const char *e = getenv("FFWD_CUDA_BERT_DELTA16");
        if (e && (!strcmp(e, "1") || !strcmp(e, "on") || !strcmp(e, "force")))
            mode = 1;
        else if (e && (!strcmp(e, "0") || !strcmp(e, "off")))
            mode = -1;
        init = 1;
    }
    return mode;
}

// cuBLASLt fused bias/GeLU epilogue for the BERT FFN gate-up GEMM. Default on
// for the exact-GeLU bf16/f16 BERT path; 0/off forces the GEMM + separate
// bias-GeLU kernel back for A/B. It routes the gate-up projection through
// cublasLtMatmul with a fused bias + erf-GeLU epilogue, replacing the separate
// bias-GeLU kernel and removing the FFN intermediate HBM round-trip.
static int bert_cublaslt_mode(void) {
    static int init = 0, mode = 0;
    if (!init) {
        const char *e = getenv("FFWD_CUDA_CUBLASLT");
        if (e && (!strcmp(e, "1") || !strcmp(e, "on") || !strcmp(e, "force")))
            mode = 1;
        else if (e && (!strcmp(e, "0") || !strcmp(e, "off")))
            mode = -1;
        init = 1;
    }
    return mode;
}

// BERT QKV projection storage. 0/default gates on total tokens; 1/on forces the
// cuBLASLt qkv BIAS epilogue to write QKV in the active 16-bit dtype; 0/off keeps
// the current F32 QKV tensor for A/B. This only applies to the BERT flash path.
static int bert_qkv16_mode(void) {
    static int init = 0, mode = 0;
    if (!init) {
        const char *e = getenv("FFWD_CUDA_BERT_QKV16");
        if (e && (!strcmp(e, "1") || !strcmp(e, "on") || !strcmp(e, "force")))
            mode = 1;
        else if (e && (!strcmp(e, "0") || !strcmp(e, "off")))
            mode = -1;
        init = 1;
    }
    return mode;
}

// BERT flash K/V-staging pipeline. Diagnostic switch over the single release
// path: 0/default rides the residual16 gate (the shipped behavior), 1/on forces
// the cp.async pipeline on, 0/off forces the legacy single-buffer staging. Used
// to A/B the pipeline at one token count without rebuilding while the
// non-residual16 bf16 correctness question is open.
static int bert_flash_pipeline_mode(void) {
    static int init = 0, mode = 0;
    if (!init) {
        const char *e = getenv("FFWD_CUDA_BERT_FLASH_PIPELINE");
        if (e && (!strcmp(e, "1") || !strcmp(e, "on") || !strcmp(e, "force")))
            mode = 1;
        else if (e && (!strcmp(e, "0") || !strcmp(e, "off")))
            mode = -1;
        init = 1;
    }
    return mode;
}

static int cuda_forward_batch_bert(ffwd_cuda_ctx_t *ctx, const ffwd_input_t *inputs, int batch) {
    const ffwd_config_t *c = &ctx->config;
    int total = 0;
    if (cuda_upload_packed_inputs(ctx, inputs, batch, &total) != 0)
        return -1;

    int hidden = c->hidden_size;
    int inter = c->intermediate_size;
    int qkv_dim = c->q_dim + 2 * c->kv_dim;
    int q_offset = 0, k_offset = c->q_dim, v_offset = c->q_dim + c->kv_dim;
    float scale = 1.0f / sqrtf((float)c->head_dim);
    float eps = c->layer_norm_eps;
    int threads = 256;
    size_t ln_smem = (size_t)hidden * sizeof(float);

    // Reduced-precision activation path. The same plan is used by
    // ensure_buffers, so the forward loop only references scratch that was
    // allocated for this exact path.
    bert_cuda_plan_t bp;
    bert_cuda_plan(ctx, total, &bp);
    int flash = bp.flash;
    int bf16_act = bp.bf16_act;
    cuda_dtype_t act_dtype = bp.act_dtype;
    void *act16 = act_dtype == CUDA_DTYPE_F16 ? ctx->act_f16
                                              : (act_dtype == CUDA_DTYPE_BF16 ? ctx->act_bf16 : NULL);
    void *resid16 = act_dtype == CUDA_DTYPE_F16 ? ctx->x_f16
                                                : (act_dtype == CUDA_DTYPE_BF16 ? ctx->x_bf16 : NULL);
    __nv_bfloat16 *act_bf16 = (__nv_bfloat16 *)ctx->act_bf16;
    __nv_bfloat16 *resid_bf16 = (__nv_bfloat16 *)ctx->x_bf16;
    __half *act_f16 = (__half *)ctx->act_f16;
    __half *resid_f16 = (__half *)ctx->x_f16;
    int bf16_resid = bp.bf16_resid;
    int f16_resid = bp.f16_resid;
    int use_resid16 = bp.use_resid16;
    int ffn16 = bp.ffn16;
    int delta16 = bp.delta16;
    int cublaslt_avail = bp.cublaslt_avail;
    int cublaslt = bp.cublaslt;
    int qkv16 = bp.qkv16;

    if (bf16_resid) {
        bert_embed_layer_norm_bf16_kernel<<<total, 256, ln_smem, ctx->stream>>>(
            resid_bf16, ctx->token_ids, ctx->positions, ctx->embed_tokens.d, ctx->embed_tokens.dtype,
            ctx->position_embeddings, ctx->token_type_embedding, ctx->ffwd_ln_w, ctx->ffwd_ln_b,
            total, hidden, c->vocab_size, eps);
    } else if (f16_resid) {
        bert_embed_layer_norm_f16_kernel<<<total, 256, ln_smem, ctx->stream>>>(
            resid_f16, ctx->token_ids, ctx->positions, ctx->embed_tokens.d, ctx->embed_tokens.dtype,
            ctx->position_embeddings, ctx->token_type_embedding, ctx->ffwd_ln_w, ctx->ffwd_ln_b,
            total, hidden, c->vocab_size, eps);
    } else if (act_dtype == CUDA_DTYPE_F16) {
        bert_embed_layer_norm_f16_act_kernel<<<total, 256, ln_smem, ctx->stream>>>(
            ctx->x, act_f16, ctx->token_ids, ctx->positions, ctx->embed_tokens.d,
            ctx->embed_tokens.dtype, ctx->position_embeddings, ctx->token_type_embedding,
            ctx->ffwd_ln_w, ctx->ffwd_ln_b, total, hidden, c->vocab_size, eps);
    } else {
        bert_embed_layer_norm_kernel<<<total, 256, ln_smem, ctx->stream>>>(
            ctx->x, bf16_act ? act_bf16 : nullptr, ctx->token_ids, ctx->positions,
            ctx->embed_tokens.d, ctx->embed_tokens.dtype, ctx->position_embeddings,
            ctx->token_type_embedding, ctx->ffwd_ln_w, ctx->ffwd_ln_b, total, hidden, c->vocab_size,
            eps);
    }
    if (launch_check() != 0)
        return -1;

    // The materialized BERT fallback uses F32 attention here. BERT flash keeps
    // scores in registers and does not use the pointer-array batched GEMM setup.
    if (!flash && attn_batched_setup(ctx, batch, q_offset, qkv_dim, CUDA_DTYPE_F32) != 0)
        return -1;

    for (int layer = 0; layer < c->n_layers; layer++) {
        cuda_layer_t *l = &ctx->layers[layer];
        const void *norm_act = use_resid16 ? resid16 : act16;

        // Self-attention (post-norm): QKV read x directly, then +bias. cuBLASLt
        // fuses the qkv bias into the GEMM epilogue. The large-batch BERT flash
        // path can write QKV straight to 16-bit storage, so K/V expansion and Q
        // staging do not reread a full F32 [tokens x qkv_dim] tensor.
        void *qkv_src = ctx->qkv;
        cuda_dtype_t qkv_dtype = CUDA_DTYPE_F32;
        if (qkv16) {
            qkv_src = ctx->qkv_16;
            qkv_dtype = act_dtype;
            if (linear_lt(ctx, &l->qkv, norm_act, qkv_src, act_dtype, l->qkv_bias,
                          CUBLASLT_EPILOGUE_BIAS, total, hidden, qkv_dim, hidden) != 0)
                return -1;
        } else if (cublaslt_avail) {
            if (linear_lt(ctx, &l->qkv, norm_act, ctx->qkv, CUDA_DTYPE_F32, l->qkv_bias,
                          CUBLASLT_EPILOGUE_BIAS, total, hidden, qkv_dim, hidden) != 0)
                return -1;
        } else {
            if (act_dtype == CUDA_DTYPE_BF16) {
                if (linear_bf16x(ctx, &l->qkv, norm_act, ctx->qkv, total, hidden, qkv_dim, hidden,
                                 0.0f) != 0)
                    return -1;
            } else if (act_dtype == CUDA_DTYPE_F16) {
                if (linear_f16x(ctx, &l->qkv, norm_act, ctx->qkv, total, hidden, qkv_dim, hidden,
                                0.0f) != 0)
                    return -1;
            } else {
                if (linear(ctx, &l->qkv, ctx->x, ctx->qkv, total, hidden, qkv_dim) != 0)
                    return -1;
            }
            int qkv_count = total * qkv_dim;
            add_row_bias_kernel<<<(qkv_count + 255) / 256, 256, 0, ctx->stream>>>(
                ctx->qkv, l->qkv_bias, total, qkv_dim);
            if (launch_check() != 0)
                return -1;
        }
        // Flash writes reduced-precision attention output into act16; the
        // materialized path writes F32 attn_out. The K/V-staging pipeline rides
        // the residual16 gate by default; the diagnostic env knob forces it
        // on/off to validate that single release path.
        int pipe_mode = bert_flash_pipeline_mode();
        int bert_pipeline = pipe_mode > 0 ? 1 : (pipe_mode < 0 ? 0 : use_resid16);
        if (cuda_attention_gemm(ctx, ctx->offsets_host, batch, q_offset, k_offset, v_offset, qkv_dim,
                                qkv_src, qkv_dtype, scale, flash ? act16 : NULL,
                                flash ? act_dtype : CUDA_DTYPE_F32, bert_pipeline) != 0)
            return -1;

        // Output dense(+bias), residual into x (F32), then post-attention LayerNorm
        // (also emitting bf16 act for the FFN gate-up GEMM in the bf16 path). With
        // flash the attn output is already bf16 in `act`, so wo reads it directly.
        if (flash && act_dtype == CUDA_DTYPE_BF16) {
            if (delta16) {
                if (linear_16x_to_16(ctx, &l->wo, act_bf16, CUDA_DTYPE_BF16, ctx->resid_delta_16,
                                     CUDA_DTYPE_BF16, total, c->q_dim, hidden, c->q_dim) != 0)
                    return -1;
            } else if (linear_bf16x(ctx, &l->wo, act_bf16, ctx->x, total, c->q_dim, hidden, c->q_dim,
                                    use_resid16 ? 0.0f : 1.0f) != 0) {
                return -1;
            }
        } else if (flash && act_dtype == CUDA_DTYPE_F16) {
            if (delta16) {
                if (linear_16x_to_16(ctx, &l->wo, act_f16, CUDA_DTYPE_F16, ctx->resid_delta_16,
                                     CUDA_DTYPE_F16, total, c->q_dim, hidden, c->q_dim) != 0)
                    return -1;
            } else if (linear_f16x(ctx, &l->wo, act_f16, ctx->x, total, c->q_dim, hidden, c->q_dim,
                                   use_resid16 ? 0.0f : 1.0f) != 0) {
                return -1;
            }
        } else if (linear_accum(ctx, &l->wo, ctx->attn_out, ctx->x, total, c->q_dim, hidden) != 0) {
            return -1;
        }
        if (bf16_resid && delta16) {
            layer_norm_bias_resid_bf16_delta16_kernel<<<total, 256, ln_smem, ctx->stream>>>(
                resid_bf16, resid_bf16, (const __nv_bfloat16 *)ctx->resid_delta_16, l->o_bias,
                l->input_norm, l->attn_ln_bias, total, hidden, eps);
        } else if (bf16_resid) {
            layer_norm_bias_resid_bf16_kernel<<<total, 256, ln_smem, ctx->stream>>>(
                resid_bf16, resid_bf16, ctx->x, l->o_bias, l->input_norm, l->attn_ln_bias, total,
                hidden, eps);
        } else if (f16_resid && delta16) {
            layer_norm_bias_resid_f16_delta16_kernel<<<total, 256, ln_smem, ctx->stream>>>(
                resid_f16, resid_f16, (const __half *)ctx->resid_delta_16, l->o_bias, l->input_norm,
                l->attn_ln_bias, total, hidden, eps);
        } else if (f16_resid) {
            layer_norm_bias_resid_f16_kernel<<<total, 256, ln_smem, ctx->stream>>>(
                resid_f16, resid_f16, ctx->x, l->o_bias, l->input_norm, l->attn_ln_bias, total,
                hidden, eps);
        } else if (act_dtype == CUDA_DTYPE_F16) {
            layer_norm_bias_f16_kernel<<<total, 256, ln_smem, ctx->stream>>>(
                ctx->x, act_f16, ctx->x, l->o_bias, l->input_norm, l->attn_ln_bias, total, hidden,
                eps);
        } else {
            layer_norm_bias_kernel<<<total, 256, ln_smem, ctx->stream>>>(
                ctx->x, bf16_act ? act_bf16 : nullptr, ctx->x, l->o_bias, l->input_norm,
                l->attn_ln_bias, total, hidden, eps);
        }
        if (launch_check() != 0)
            return -1;
        norm_act = use_resid16 ? resid16 : act16;

        // Feed-forward (post-norm): dense(+bias) -> GeLU -> dense(+bias). cuBLASLt
        // fuses bias + erf-GeLU into the gate-up GEMM epilogue, so no separate
        // bias/activation kernel runs. The fused output goes to its own
        // ffn_gate_up_16 buffer, NOT act16: when residual16 is off, norm_act IS
        // act16, and a GEMM whose output overlaps its input is undefined (it read
        // and overwrote the same storage, nondeterministically corrupting rows).
        // The down projection reads this intermediate below.
        if (cublaslt) {
            if (linear_lt(ctx, &l->gate_up_proj, norm_act, ctx->ffn_gate_up_16, act_dtype,
                          l->ffn_inter_bias, CUBLASLT_EPILOGUE_GELU_BIAS, total, hidden, inter,
                          hidden) != 0)
                return -1;
        } else {
            if (act_dtype == CUDA_DTYPE_BF16) {
                if (ffn16) {
                    if (linear_16x_to_16(ctx, &l->gate_up_proj, norm_act, CUDA_DTYPE_BF16,
                                         ctx->ffn_gate_up_16, CUDA_DTYPE_BF16, total, hidden, inter,
                                         hidden) != 0)
                        return -1;
                } else if (linear_bf16x(ctx, &l->gate_up_proj, norm_act, ctx->ffn_gate_up, total,
                                        hidden, inter, hidden, 0.0f) != 0) {
                    return -1;
                }
            } else if (act_dtype == CUDA_DTYPE_F16) {
                if (ffn16) {
                    if (linear_16x_to_16(ctx, &l->gate_up_proj, norm_act, CUDA_DTYPE_F16,
                                         ctx->ffn_gate_up_16, CUDA_DTYPE_F16, total, hidden, inter,
                                         hidden) != 0)
                        return -1;
                } else if (linear_f16x(ctx, &l->gate_up_proj, norm_act, ctx->ffn_gate_up, total,
                                       hidden, inter, hidden, 0.0f) != 0) {
                    return -1;
                }
            } else {
                if (linear(ctx, &l->gate_up_proj, ctx->x, ctx->ffn_gate_up, total, hidden, inter) !=
                    0)
                    return -1;
            }
            int inter_count = total * inter;
            if (act_dtype == CUDA_DTYPE_BF16) {
                if (ffn16) {
                    const __nv_bfloat16 *gate = (const __nv_bfloat16 *)ctx->ffn_gate_up_16;
                    if (c->ffn_act == FFWD_ACT_GELU_TANH)
                        bias_gelu_tanh_16_to_16_kernel<<<(inter_count + threads - 1) / threads,
                                                         threads, 0, ctx->stream>>>(
                            act_bf16, gate, l->ffn_inter_bias, total, inter);
                    else
                        bias_gelu_16_to_16_kernel<<<(inter_count + threads - 1) / threads, threads, 0,
                                                    ctx->stream>>>(act_bf16, gate, l->ffn_inter_bias,
                                                                   total, inter);
                } else if (c->ffn_act == FFWD_ACT_GELU_TANH) {
                    bias_gelu_tanh_to_bf16_kernel<<<(inter_count + threads - 1) / threads, threads, 0,
                                                    ctx->stream>>>(act_bf16, ctx->ffn_gate_up,
                                                                   l->ffn_inter_bias, total, inter);
                } else {
                    bias_gelu_to_bf16_kernel<<<(inter_count + threads - 1) / threads, threads, 0,
                                               ctx->stream>>>(act_bf16, ctx->ffn_gate_up,
                                                              l->ffn_inter_bias, total, inter);
                }
            } else if (act_dtype == CUDA_DTYPE_F16) {
                if (ffn16) {
                    const __half *gate = (const __half *)ctx->ffn_gate_up_16;
                    if (c->ffn_act == FFWD_ACT_GELU_TANH)
                        bias_gelu_tanh_16_to_16_kernel<<<(inter_count + threads - 1) / threads,
                                                         threads, 0, ctx->stream>>>(
                            act_f16, gate, l->ffn_inter_bias, total, inter);
                    else
                        bias_gelu_16_to_16_kernel<<<(inter_count + threads - 1) / threads, threads, 0,
                                                    ctx->stream>>>(act_f16, gate, l->ffn_inter_bias,
                                                                   total, inter);
                } else if (c->ffn_act == FFWD_ACT_GELU_TANH) {
                    bias_gelu_tanh_to_f16_kernel<<<(inter_count + threads - 1) / threads, threads, 0,
                                                   ctx->stream>>>(act_f16, ctx->ffn_gate_up,
                                                                  l->ffn_inter_bias, total, inter);
                } else {
                    bias_gelu_to_f16_kernel<<<(inter_count + threads - 1) / threads, threads, 0,
                                              ctx->stream>>>(act_f16, ctx->ffn_gate_up,
                                                             l->ffn_inter_bias, total, inter);
                }
            } else if (c->ffn_act == FFWD_ACT_GELU_TANH) {
                bias_gelu_tanh_kernel<<<(inter_count + threads - 1) / threads, threads, 0,
                                        ctx->stream>>>(ctx->ffn_gate_up, l->ffn_inter_bias, total,
                                                       inter);
            } else {
                bias_gelu_kernel<<<(inter_count + threads - 1) / threads, threads, 0, ctx->stream>>>(
                    ctx->ffn_gate_up, l->ffn_inter_bias, total, inter);
            }
        }
        if (launch_check() != 0)
            return -1;
        // Post-GELU FFN intermediate source for the down projection: the cuBLASLt
        // epilogue wrote it to the dedicated ffn_gate_up_16 buffer; the
        // separate-kernel path left it in act16.
        const __nv_bfloat16 *ffn_inter_bf16 =
            (const __nv_bfloat16 *)(cublaslt ? ctx->ffn_gate_up_16 : (void *)act_bf16);
        const __half *ffn_inter_f16 =
            (const __half *)(cublaslt ? ctx->ffn_gate_up_16 : (void *)act_f16);
        if (act_dtype == CUDA_DTYPE_BF16) {
            if (delta16) {
                if (linear_16x_to_16(ctx, &l->down_proj, ffn_inter_bf16, CUDA_DTYPE_BF16,
                                     ctx->resid_delta_16, CUDA_DTYPE_BF16, total, inter, hidden,
                                     inter) != 0)
                    return -1;
            } else if (linear_bf16x(ctx, &l->down_proj, ffn_inter_bf16, ctx->x, total, inter, hidden,
                                    inter, use_resid16 ? 0.0f : 1.0f) != 0) {
                return -1;
            }
        } else if (act_dtype == CUDA_DTYPE_F16) {
            if (delta16) {
                if (linear_16x_to_16(ctx, &l->down_proj, ffn_inter_f16, CUDA_DTYPE_F16,
                                     ctx->resid_delta_16, CUDA_DTYPE_F16, total, inter, hidden,
                                     inter) != 0)
                    return -1;
            } else if (linear_f16x(ctx, &l->down_proj, ffn_inter_f16, ctx->x, total, inter, hidden,
                                   inter, use_resid16 ? 0.0f : 1.0f) != 0) {
                return -1;
            }
        } else {
            if (linear_accum(ctx, &l->down_proj, ctx->ffn_gate_up, ctx->x, total, inter, hidden) != 0)
                return -1;
        }
        if (bf16_resid && delta16) {
            layer_norm_bias_resid_bf16_delta16_kernel<<<total, 256, ln_smem, ctx->stream>>>(
                resid_bf16, resid_bf16, (const __nv_bfloat16 *)ctx->resid_delta_16, l->ffn_out_bias,
                l->post_attn_norm, l->ffn_ln_bias, total, hidden, eps);
        } else if (bf16_resid) {
            layer_norm_bias_resid_bf16_kernel<<<total, 256, ln_smem, ctx->stream>>>(
                resid_bf16, resid_bf16, ctx->x, l->ffn_out_bias, l->post_attn_norm, l->ffn_ln_bias,
                total, hidden, eps);
        } else if (f16_resid && delta16) {
            layer_norm_bias_resid_f16_delta16_kernel<<<total, 256, ln_smem, ctx->stream>>>(
                resid_f16, resid_f16, (const __half *)ctx->resid_delta_16, l->ffn_out_bias,
                l->post_attn_norm, l->ffn_ln_bias, total, hidden, eps);
        } else if (f16_resid) {
            layer_norm_bias_resid_f16_kernel<<<total, 256, ln_smem, ctx->stream>>>(
                resid_f16, resid_f16, ctx->x, l->ffn_out_bias, l->post_attn_norm, l->ffn_ln_bias,
                total, hidden, eps);
        } else if (act_dtype == CUDA_DTYPE_F16) {
            layer_norm_bias_f16_kernel<<<total, 256, ln_smem, ctx->stream>>>(
                ctx->x, act_f16, ctx->x, l->ffn_out_bias, l->post_attn_norm, l->ffn_ln_bias, total,
                hidden, eps);
        } else {
            layer_norm_bias_kernel<<<total, 256, ln_smem, ctx->stream>>>(
                ctx->x, bf16_act ? act_bf16 : nullptr, ctx->x, l->ffn_out_bias, l->post_attn_norm,
                l->ffn_ln_bias, total, hidden, eps);
        }
        if (launch_check() != 0)
            return -1;
    }
    if (bf16_resid) {
        size_t n = (size_t)total * (size_t)hidden;
        cast_bf16_to_f32_kernel<<<(n + threads - 1) / threads, threads, 0, ctx->stream>>>(
            ctx->x, resid_bf16, n);
        if (launch_check() != 0)
            return -1;
    } else if (f16_resid) {
        size_t n = (size_t)total * (size_t)hidden;
        cast_f16_to_f32_kernel<<<(n + threads - 1) / threads, threads, 0, ctx->stream>>>(
            ctx->x, resid_f16, n);
        if (launch_check() != 0)
            return -1;
    }
    // No final norm; ffwd_cuda_encode_batch pools ctx->x directly.
    return 0;
}

static int cuda_forward_batch(ffwd_cuda_ctx_t *ctx, const ffwd_input_t *inputs, int batch) {
    if (!ctx || !inputs || batch <= 0)
        return -1;
    const ffwd_config_t *c = &ctx->config;
    if (c->family == FFWD_FAMILY_BERT)
        return cuda_forward_batch_bert(ctx, inputs, batch);
    int total = 0;
    if (cuda_upload_packed_inputs(ctx, inputs, batch, &total) != 0)
        return -1;

    int threads = 256;
    int blocks_hidden = (total * c->hidden_size + threads - 1) / threads;
    ffwd_lookup_kernel<<<blocks_hidden, threads, 0, ctx->stream>>>(
        ctx->x, ctx->token_ids, ctx->embed_tokens.d, ctx->embed_tokens.dtype, total, c->hidden_size,
        c->vocab_size);
    if (launch_check() != 0)
        return -1;

    float scale = 1.0f / sqrtf((float)c->head_dim);
    int q_offset = 0;
    int k_offset = c->q_dim;
    int v_offset = c->q_dim + c->kv_dim;
    int qkv_dim = c->q_dim + 2 * c->kv_dim;
    // With BF16 weights every producer (norm, SiLU, attention @V GEMM)
    // stores its result as BF16 directly into ctx->act_bf16, which the next
    // projection GEMM consumes before any later producer reuses the buffer
    // (single stream). No cast launches remain in the layer loop.
    __nv_bfloat16 *act = (__nv_bfloat16 *)ctx->act_bf16;
    // BF16-weight layers run attention in BF16 (cuda_attention_gemm gets `act`);
    // the batched layout must match. Weight dtype is uniform across layers.
    if (attn_batched_setup(ctx, batch, q_offset, qkv_dim,
                           ctx->weights_bf16 ? CUDA_DTYPE_BF16 : CUDA_DTYPE_F32) != 0)
        return -1;
    for (int layer = 0; layer < c->n_layers; layer++) {
        cuda_layer_t *l = &ctx->layers[layer];
        int wbf16 = l->qkv.bf16;
        if (wbf16) {
            rms_norm_kernel<<<total, 256, 0, ctx->stream>>>(act, ctx->x, l->input_norm, total,
                                                            c->hidden_size, c->rms_norm_eps);
            if (launch_check() != 0)
                return -1;
            if (linear_bf16x(ctx, &l->qkv, act, ctx->qkv, total, c->hidden_size, qkv_dim,
                             c->hidden_size, 0.0f) != 0)
                return -1;
        } else {
            rms_norm_kernel<<<total, 256, 0, ctx->stream>>>(ctx->x_norm, ctx->x, l->input_norm, total,
                                                            c->hidden_size, c->rms_norm_eps);
            if (launch_check() != 0)
                return -1;
            if (linear(ctx, &l->qkv, ctx->x_norm, ctx->qkv, total, c->hidden_size, qkv_dim) != 0)
                return -1;
        }

        if (c->qkv_bias) {
            int count = total * qkv_dim;
            add_row_bias_kernel<<<(count + 255) / 256, 256, 0, ctx->stream>>>(ctx->qkv, l->qkv_bias,
                                                                              total, qkv_dim);
            if (launch_check() != 0)
                return -1;
        }

        rms_norm_rope_qk_kernel<<<dim3(total, c->n_heads + c->n_kv_heads), 128, 0, ctx->stream>>>(
            ctx->qkv, l->q_norm, l->k_norm, ctx->positions, ctx->rope_cos, ctx->rope_sin, total,
            c->n_heads, c->n_kv_heads, c->head_dim, qkv_dim, q_offset, k_offset, c->qk_norm,
            c->rms_norm_eps);
        if (launch_check() != 0)
            return -1;

        if (cuda_attention_gemm(ctx, ctx->offsets_host, batch, q_offset, k_offset, v_offset, qkv_dim,
                                ctx->qkv, CUDA_DTYPE_F32, scale, wbf16 ? act : NULL,
                                wbf16 ? CUDA_DTYPE_BF16 : CUDA_DTYPE_F32, 0) != 0)
            return -1;

        if (wbf16) {
            if (linear_bf16x(ctx, &l->wo, act, ctx->x, total, c->q_dim, c->hidden_size, c->q_dim,
                             1.0f) != 0)
                return -1;
        } else {
            if (linear_accum(ctx, &l->wo, ctx->attn_out, ctx->x, total, c->q_dim, c->hidden_size) !=
                0)
                return -1;
        }

        int inter_count = total * c->intermediate_size;
        if (wbf16) {
            rms_norm_kernel<<<total, 256, 0, ctx->stream>>>(act, ctx->x, l->post_attn_norm, total,
                                                            c->hidden_size, c->rms_norm_eps);
            if (launch_check() != 0)
                return -1;
            if (linear_bf16x(ctx, &l->gate_up_proj, act, ctx->ffn_gate_up, total, c->hidden_size,
                             2 * c->intermediate_size, c->hidden_size, 0.0f) != 0)
                return -1;
            silu_mul_packed_kernel<<<(inter_count + threads - 1) / threads, threads, 0,
                                     ctx->stream>>>(act, c->intermediate_size, ctx->ffn_gate_up,
                                                    total, c->intermediate_size);
            if (launch_check() != 0)
                return -1;
            if (linear_bf16x(ctx, &l->down_proj, act, ctx->x, total, c->intermediate_size,
                             c->hidden_size, c->intermediate_size, 1.0f) != 0)
                return -1;
        } else {
            rms_norm_kernel<<<total, 256, 0, ctx->stream>>>(ctx->x_norm, ctx->x, l->post_attn_norm,
                                                            total, c->hidden_size, c->rms_norm_eps);
            if (launch_check() != 0)
                return -1;
            if (linear(ctx, &l->gate_up_proj, ctx->x_norm, ctx->ffn_gate_up, total, c->hidden_size,
                       2 * c->intermediate_size) != 0)
                return -1;
            silu_mul_packed_kernel<<<(inter_count + threads - 1) / threads, threads, 0,
                                     ctx->stream>>>(ctx->ffn_gate_up, 2 * c->intermediate_size,
                                                    ctx->ffn_gate_up, total, c->intermediate_size);
            if (launch_check() != 0)
                return -1;
            if (linear_ex(ctx, &l->down_proj, ctx->ffn_gate_up, CUDA_DTYPE_F32, ctx->x, total,
                          c->intermediate_size, c->hidden_size, 2 * c->intermediate_size, 1.0f) != 0)
                return -1;
        }
    }

    rms_norm_kernel<<<total, 256, 0, ctx->stream>>>(ctx->x, ctx->x, ctx->norm, total, c->hidden_size,
                                                    c->rms_norm_eps);
    if (launch_check() != 0)
        return -1;

    return 0;
}

int ffwd_cuda_encode_batch(ffwd_cuda_ctx_t *ctx,
                           const ffwd_input_t *inputs,
                           int batch,
                           float *out_embeddings) {
    if (!ctx || !inputs || batch <= 0 || !out_embeddings)
        return -1;
    const ffwd_config_t *c = &ctx->config;
    if (cuda_forward_batch(ctx, inputs, batch) != 0)
        return -1;
    if (ensure_pooled_rows(ctx, batch) != 0)
        return -1;
    if (c->pooling_mode == FFWD_POOL_LAST_TOKEN)
        last_pool_kernel<<<batch, 256, 0, ctx->stream>>>(ctx->pooled_out, ctx->x, ctx->offsets, batch,
                                                         c->hidden_size);
    else if (c->pooling_mode == FFWD_POOL_CLS)
        first_pool_kernel<<<batch, 256, 0, ctx->stream>>>(ctx->pooled_out, ctx->x, ctx->offsets,
                                                          batch, c->hidden_size);
    else
        mean_pool_kernel<<<batch, 256, 0, ctx->stream>>>(ctx->pooled_out, ctx->x, ctx->offsets, batch,
                                                         c->hidden_size);
    if (launch_check() != 0)
        return -1;
    cudaError_t ce = cudaMemcpyAsync(out_embeddings, ctx->pooled_out,
                                     (size_t)batch * c->hidden_size * sizeof(float),
                                     cudaMemcpyDeviceToHost, ctx->stream);
    if (ce == cudaSuccess)
        ce = cudaStreamSynchronize(ctx->stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "cuda output copy failed: %s\n", cudaGetErrorString(ce));
        return -1;
    }
    if (c->normalize_embeddings) {
        for (int b = 0; b < batch; b++) {
            if (ffwd_l2_normalize(out_embeddings + (size_t)b * c->hidden_size, c->hidden_size) != 0)
                return -1;
        }
    }
    return 0;
}

int ffwd_cuda_encode_spans_batch(ffwd_cuda_ctx_t *ctx,
                                 const ffwd_context_input_t *inputs,
                                 int batch,
                                 float *out_embeddings) {
    if (!ctx || !inputs || batch <= 0 || !out_embeddings)
        return -1;
    const ffwd_config_t *c = &ctx->config;
    int total_spans = 0;
    int total_tokens = 0;
    for (int b = 0; b < batch; b++) {
        const ffwd_context_input_t *input = &inputs[b];
        int n_tokens = input->input.n_tokens;
        if (!input->input.ids || n_tokens <= 0 || !input->spans || input->n_spans <= 0 ||
            total_spans > INT_MAX - input->n_spans || total_tokens > INT_MAX - n_tokens)
            return -1;
        for (int s = 0; s < input->n_spans; s++) {
            int start = input->spans[s].start;
            int len = input->spans[s].n_tokens;
            if (start < 0 || len <= 0 || start > n_tokens || len > n_tokens - start)
                return -1;
        }
        total_spans += input->n_spans;
        total_tokens += n_tokens;
    }

    ffwd_input_t *packed = (ffwd_input_t *)malloc((size_t)batch * sizeof(*packed));
    int *h_starts = (int *)malloc((size_t)total_spans * sizeof(int));
    int *h_lens = (int *)malloc((size_t)total_spans * sizeof(int));
    if (!packed || !h_starts || !h_lens) {
        free(packed);
        free(h_starts);
        free(h_lens);
        return -1;
    }

    int token_offset = 0;
    int span_offset = 0;
    for (int b = 0; b < batch; b++) {
        const ffwd_context_input_t *input = &inputs[b];
        packed[b] = input->input;
        for (int s = 0; s < input->n_spans; s++) {
            h_starts[span_offset] = token_offset + input->spans[s].start;
            h_lens[span_offset] = input->spans[s].n_tokens;
            span_offset++;
        }
        token_offset += input->input.n_tokens;
    }

    int rc = -1;
    cudaError_t ce;
    if (cuda_forward_batch(ctx, packed, batch) != 0)
        goto cleanup;
    if (ensure_pooled_rows(ctx, total_spans) != 0 || ensure_span_buffers(ctx, total_spans) != 0)
        goto cleanup;
    ce = cudaMemcpyAsync(ctx->span_starts, h_starts, (size_t)total_spans * sizeof(int),
                         cudaMemcpyHostToDevice, ctx->stream);
    if (ce == cudaSuccess)
        ce = cudaMemcpyAsync(ctx->span_lens, h_lens, (size_t)total_spans * sizeof(int),
                             cudaMemcpyHostToDevice, ctx->stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "cuda span copy failed: %s\n", cudaGetErrorString(ce));
        goto cleanup;
    }
    span_pool_kernel<<<total_spans, 256, 0, ctx->stream>>>(
        ctx->pooled_out, ctx->x, ctx->span_starts, ctx->span_lens, total_spans, c->hidden_size);
    if (launch_check() != 0)
        goto cleanup;
    ce = cudaMemcpyAsync(out_embeddings, ctx->pooled_out,
                         (size_t)total_spans * c->hidden_size * sizeof(float), cudaMemcpyDeviceToHost,
                         ctx->stream);
    if (ce == cudaSuccess)
        ce = cudaStreamSynchronize(ctx->stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "cuda span output copy failed: %s\n", cudaGetErrorString(ce));
        goto cleanup;
    }
    rc = 0;

cleanup:
    free(h_lens);
    free(h_starts);
    free(packed);
    return rc;
}

int ffwd_cuda_encode_into(ffwd_cuda_ctx_t *ctx,
                          const int *token_ids,
                          int n_tokens,
                          float *out_embedding) {
    ffwd_input_t input = {token_ids, n_tokens};
    return ffwd_cuda_encode_batch(ctx, &input, 1, out_embedding);
}

float *ffwd_cuda_encode(ffwd_cuda_ctx_t *ctx, const int *token_ids, int n_tokens) {
    if (!ctx)
        return NULL;
    int hidden = ctx->config.hidden_size;
    float *out = (float *)malloc((size_t)hidden * sizeof(float));
    if (!out)
        return NULL;
    if (ffwd_cuda_encode_into(ctx, token_ids, n_tokens, out) != 0) {
        free(out);
        return NULL;
    }
    return out;
}

/* ========================================================================
 * Late-interaction CUDA path
 *
 * The transformer forward and the 1_Dense projection run on the GPU; MaxSim
 * scoring stays on the host, where the grouped-GEMM scorer is far faster than a
 * device graph for this small op. So the encoders copy the projected token
 * vectors back to host memory and the caller runs ffwd_late_maxsim_batch there.
 * ======================================================================== */

struct ffwd_cuda_late_ctx {
    ffwd_cuda_ctx_t *base;       // device base model; does not own its CPU model
    ffwd_late_model_t *cpu_late; // owns the CPU base model + projection host data
    cuda_matrix_t projection;    // device [token_dim, hidden]
    float *proj_dev;             // device scratch [total_tokens, token_dim], grown on demand
    size_t proj_cap;             // capacity of proj_dev in floats
    int token_dim;
};

ffwd_cuda_late_ctx_t *ffwd_cuda_late_load(const char *model_dir) {
    ffwd_late_model_t *cpu_late = ffwd_late_model_load(model_dir);
    if (!cpu_late)
        return NULL;

    ffwd_cuda_late_ctx_t *ctx = (ffwd_cuda_late_ctx_t *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        ffwd_late_model_free(cpu_late);
        return NULL;
    }
    ctx->cpu_late = cpu_late;
    ctx->token_dim = ffwd_late_model_token_dim(cpu_late);

    // Upload the base transformer weights; the CPU late model keeps ownership.
    ctx->base = cuda_ctx_from_model(ffwd_late_model_base(cpu_late), 0);
    if (!ctx->base) {
        ffwd_cuda_late_free(ctx);
        return NULL;
    }

    // Upload the 1_Dense projection [token_dim, hidden].
    const ffwd_weight_ref_t *proj = ffwd_late_model_projection(cpu_late);
    if (!proj || ctx->token_dim <= 0 ||
        load_matrix(&ctx->projection, proj, ctx->token_dim, ctx->base->config.hidden_size,
                    ctx->base->weight_dtype) != 0) {
        ffwd_cuda_late_free(ctx);
        return NULL;
    }
    return ctx;
}

void ffwd_cuda_late_free(ffwd_cuda_late_ctx_t *ctx) {
    if (!ctx)
        return;
    cuda_matrix_free(&ctx->projection);
    cudaFree(ctx->proj_dev);
    ffwd_cuda_free(ctx->base);           // own_cpu = 0, so this keeps the CPU model
    ffwd_late_model_free(ctx->cpu_late); // frees the CPU base model + projection
    free(ctx);
}

const ffwd_config_t *ffwd_cuda_late_config(const ffwd_cuda_late_ctx_t *ctx) {
    return ctx ? &ctx->base->config : NULL;
}

int ffwd_cuda_late_token_dim(const ffwd_cuda_late_ctx_t *ctx) { return ctx ? ctx->token_dim : 0; }

static int ensure_late_proj(ffwd_cuda_late_ctx_t *ctx, int total) {
    size_t need = (size_t)total * (size_t)ctx->token_dim;
    if (need <= ctx->proj_cap)
        return 0;
    cudaFree(ctx->proj_dev);
    ctx->proj_dev = NULL;
    ctx->proj_cap = 0;
    if (cudaMalloc((void **)&ctx->proj_dev, need * sizeof(float)) != cudaSuccess)
        return -1;
    ctx->proj_cap = need;
    return 0;
}

/* Forward all inputs, project every token to token_dim, and copy the packed
 * [total, token_dim] F32 result to a freshly malloc'd host buffer (caller frees).
 * On success *proj_host and *total_out are set; base->offsets_host holds the
 * packed per-document token offsets. */
static int cuda_late_forward_project(ffwd_cuda_late_ctx_t *ctx,
                                     const ffwd_input_t *inputs,
                                     int n_docs,
                                     float **proj_host,
                                     int *total_out) {
    ffwd_cuda_ctx_t *base = ctx->base;
    const ffwd_config_t *c = &base->config;
    if (cuda_forward_batch(base, inputs, n_docs) != 0)
        return -1;
    int total = base->offsets_host[n_docs];
    if (total <= 0 || ensure_late_proj(ctx, total) != 0)
        return -1;
    if (linear(base, &ctx->projection, base->x, ctx->proj_dev, total, c->hidden_size,
               ctx->token_dim) != 0)
        return -1;
    float *host = (float *)malloc((size_t)total * (size_t)ctx->token_dim * sizeof(float));
    if (!host)
        return -1;
    cudaError_t ce =
        cudaMemcpyAsync(host, ctx->proj_dev, (size_t)total * (size_t)ctx->token_dim * sizeof(float),
                        cudaMemcpyDeviceToHost, base->stream);
    if (ce == cudaSuccess)
        ce = cudaStreamSynchronize(base->stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "cuda late output copy failed: %s\n", cudaGetErrorString(ce));
        free(host);
        return -1;
    }
    *proj_host = host;
    *total_out = total;
    return 0;
}

int ffwd_cuda_late_encode_tokens(ffwd_cuda_late_ctx_t *ctx,
                                 const int *token_ids,
                                 int n_tokens,
                                 int normalize,
                                 float *out_vectors) {
    if (!ctx || !token_ids || n_tokens <= 0 || !out_vectors || ctx->token_dim <= 0)
        return -1;
    ffwd_input_t input = {token_ids, n_tokens};
    float *proj_host = NULL;
    int total = 0;
    if (cuda_late_forward_project(ctx, &input, 1, &proj_host, &total) != 0)
        return -1;
    memcpy(out_vectors, proj_host, (size_t)total * (size_t)ctx->token_dim * sizeof(float));
    free(proj_host);
    if (normalize) {
        for (int i = 0; i < total; i++)
            if (ffwd_l2_normalize(out_vectors + (size_t)i * ctx->token_dim, ctx->token_dim) != 0)
                return -1;
    }
    return 0;
}

int ffwd_cuda_late_encode_docs(ffwd_cuda_late_ctx_t *ctx,
                               const int *const *doc_ids,
                               const int *n_tokens,
                               const int *const *keep,
                               const int *n_keep,
                               int n_docs,
                               int normalize,
                               float *out_vectors,
                               int *out_offsets) {
    if (!ctx || !doc_ids || !n_tokens || !keep || !n_keep || n_docs <= 0 || !out_vectors ||
        !out_offsets || ctx->token_dim <= 0)
        return -1;
    int dim = ctx->token_dim;

    ffwd_input_t *inputs = (ffwd_input_t *)malloc((size_t)n_docs * sizeof(*inputs));
    if (!inputs)
        return -1;

    int total_keep = 0;
    out_offsets[0] = 0;
    for (int d = 0; d < n_docs; d++) {
        if (!doc_ids[d] || n_tokens[d] <= 0 || !keep[d] || n_keep[d] <= 0 ||
            n_keep[d] > n_tokens[d] || total_keep > INT_MAX - n_keep[d]) {
            free(inputs);
            return -1;
        }
        for (int k = 0; k < n_keep[d]; k++) {
            if (keep[d][k] < 0 || keep[d][k] >= n_tokens[d]) {
                free(inputs);
                return -1;
            }
        }
        inputs[d].ids = doc_ids[d];
        inputs[d].n_tokens = n_tokens[d];
        total_keep += n_keep[d];
        out_offsets[d + 1] = total_keep;
    }

    float *proj_host = NULL;
    int total = 0;
    if (cuda_late_forward_project(ctx, inputs, n_docs, &proj_host, &total) != 0) {
        free(inputs);
        return -1;
    }

    /* Gather each document's kept token rows (in packed forward order) into the
     * output, the layout MaxSim consumes - matching the CPU/MLX paths. */
    const int *off = ctx->base->offsets_host;
    int pos = 0;
    for (int d = 0; d < n_docs; d++) {
        int start = off[d];
        for (int k = 0; k < n_keep[d]; k++) {
            memcpy(out_vectors + (size_t)pos * dim, proj_host + (size_t)(start + keep[d][k]) * dim,
                   (size_t)dim * sizeof(float));
            pos++;
        }
    }
    free(proj_host);
    free(inputs);

    if (normalize) {
        for (int i = 0; i < total_keep; i++)
            if (ffwd_l2_normalize(out_vectors + (size_t)i * dim, dim) != 0)
                return -1;
    }
    return 0;
}
