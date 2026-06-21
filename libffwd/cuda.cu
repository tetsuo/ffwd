extern "C" {
#include "cuda.h"
#include "model_types.h"
}

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *d; // device buffer; holds __nv_bfloat16 elements when bf16
    int rows;
    int cols;
    int bf16; // 1 if d holds __nv_bfloat16, 0 if F32
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
    cudaStream_t stream;
    cublasHandle_t blas;
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
    int batch_cap;
    int max_seq_cap;
    float *x;
    float *x_norm;
    float *qkv;
    float *attn_out;
    float *ffn_gate_up;
    void *act_bf16;    // BF16 cast of a GEMM activation operand (bf16 weights)
    float *weight_f32; // F32 widen scratch for one BF16 weight (memory-only bf16)
    size_t weight_f32_elems;
    float *pooled_out;
    float *rope_cos;
    float *rope_sin;
    int *token_ids;
    int *offsets;
    int *offsets_host;
    int *positions;
    float *kexp;
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

// Store weights as native BF16 on the device when set (halves weight memory and
// feeds BF16 tensor cores); default keeps exact F32 for F32 model files. When
// the flag was never set explicitly, the loader follows the model file dtype,
// matching the CPU and MLX backends.
static int g_weights_bf16 = 0;
static int g_weights_bf16_set = 0;

int ffwd_cuda_set_weights_bf16(int on) {
    g_weights_bf16 = on ? 1 : 0;
    g_weights_bf16_set = 1;
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

// Upload a host F32 weight buffer to the device, as BF16 when g_weights_bf16 is
// set (rounding each value) or as F32 otherwise. Sets m->d, rows, cols, bf16.
static int upload_weight(cuda_matrix_t *m, const float *src, size_t count, int rows, int cols) {
    cudaError_t e;
    if (g_weights_bf16) {
        uint16_t *tmp16 = (uint16_t *)malloc(count * sizeof(uint16_t));
        if (!tmp16)
            return -1;
        for (size_t i = 0; i < count; i++)
            tmp16[i] = f32_to_bf16(src[i]);
        e = cudaMalloc((void **)&m->d, count * sizeof(__nv_bfloat16));
        if (e != cudaSuccess) {
            fprintf(stderr, "cudaMalloc weight failed: %s\n", cudaGetErrorString(e));
            free(tmp16);
            return -1;
        }
        e = cudaMemcpy(m->d, tmp16, count * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        free(tmp16);
        m->bf16 = 1;
    } else {
        e = cudaMalloc((void **)&m->d, count * sizeof(float));
        if (e != cudaSuccess) {
            fprintf(stderr, "cudaMalloc weight failed: %s\n", cudaGetErrorString(e));
            return -1;
        }
        e = cudaMemcpy(m->d, src, count * sizeof(float), cudaMemcpyHostToDevice);
        m->bf16 = 0;
    }
    if (e != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy weight failed: %s\n", cudaGetErrorString(e));
        cuda_matrix_free(m);
        return -1;
    }
    m->rows = rows;
    m->cols = cols;
    return 0;
}

static int load_matrix(cuda_matrix_t *m, const ffwd_weight_ref_t *w, int rows, int cols) {
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
    int r = upload_weight(m, tmp, count, rows, cols);
    free(tmp);
    return r;
}

static int load_qkv_matrix(cuda_matrix_t *m, const ffwd_layer_t *src, const ffwd_config_t *c) {
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
    int r = upload_weight(m, tmp, total, rows, cols);
    free(tmp);
    return r;
}

static int load_gate_up_matrix(cuda_matrix_t *m, const ffwd_layer_t *src, const ffwd_config_t *c) {
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
    int r = upload_weight(m, tmp, total, rows, cols);
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
    float *x, const int *ids, const void *emb, int emb_bf16, int total, int hidden, int vocab) {
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
    x[idx] =
        emb_bf16 ? __bfloat162float(((const __nv_bfloat16 *)emb)[off]) : ((const float *)emb)[off];
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

__device__ static inline void store_act(float *out, size_t i, float v) { out[i] = v; }

__device__ static inline void store_act(__nv_bfloat16 *out, size_t i, float v) {
    out[i] = __float2bfloat16(v);
}

__device__ static inline float load_embedding_value(const void *emb, int emb_bf16, size_t i) {
    return emb_bf16 ? __bfloat162float(((const __nv_bfloat16 *)emb)[i]) : ((const float *)emb)[i];
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
// V_T is float, or __nv_bfloat16 when V feeds a BF16 @V GEMM directly.
template <typename V_T>
__global__ static void attn_expand_kv_kernel(float *kexp,
                                             V_T *vexp,
                                             const float *qkv,
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
    const float *base = qkv + (size_t)t * qkv_dim;
    kexp[idx] = base[k_offset + kv * head_dim + e];
    store_act(vexp, idx, base[v_offset + kv * head_dim + e]);
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
    for (int d = tid; d < hidden; d += blockDim.x)
        out[(size_t)row * hidden + d] = (rowbuf[d] - mean) * inv * gamma[d] + beta[d];
}

// BERT SkipLayerNorm variant for the common post-GEMM shape. The residual has
// already been accumulated by cuBLAS beta=1, so this fuses the remaining row
// bias with LayerNorm and keeps the pre-normalized row in shared memory.
__global__ static void layer_norm_bias_kernel(float *out,
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
    for (int d = tid; d < dim; d += blockDim.x)
        o[d] = (rowbuf[d] - mean) * inv * gamma[d] + beta[d];
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

// y = w @ x (+ beta*y). With BF16 weights and an already-BF16 activation
// (x_is_bf16 - the Qwen fused path) both inputs are BF16 with F32 accumulation.
// For the BERT family the activation stays F32 and the BF16 weight is widened
// back to F32 here, so the GEMM is exact (honoring gemm_compute): BF16 is then a
// storage choice that costs no quality, which the post-norm LayerNorm needs.
// With F32 weights this is the exact path.
static int linear_ex(ffwd_cuda_ctx_t *ctx,
                     const cuda_matrix_t *w,
                     const void *x,
                     int x_is_bf16,
                     float *y,
                     int rows,
                     int in_dim,
                     int out_dim,
                     int x_stride,
                     float beta) {
    const float alpha = 1.0f;
    if (w->bf16) {
        if (!x_is_bf16 && ctx->config.family == FFWD_FAMILY_BERT) {
            size_t welems = (size_t)out_dim * (size_t)in_dim;
            if (ensure_weight_f32(ctx, welems) != 0)
                return -1;
            int wt = 256;
            cast_bf16_to_f32_kernel<<<(welems + wt - 1) / wt, wt, 0, ctx->stream>>>(
                ctx->weight_f32, (const __nv_bfloat16 *)w->d, welems);
            if (launch_check() != 0)
                return -1;
            CUBLAS_CHECK(cublasGemmEx(ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N, out_dim, rows, in_dim,
                                      &alpha, ctx->weight_f32, CUDA_R_32F, in_dim, x, CUDA_R_32F,
                                      x_stride, &beta, y, CUDA_R_32F, out_dim, gemm_compute(),
                                      CUBLAS_GEMM_DEFAULT));
            return 0;
        }
        const void *xb = x;
        if (!x_is_bf16) {
            size_t n = (size_t)rows * (size_t)x_stride;
            int threads = 256;
            cast_f32_to_bf16_kernel<<<(n + threads - 1) / threads, threads, 0, ctx->stream>>>(
                (__nv_bfloat16 *)ctx->act_bf16, (const float *)x, n);
            if (launch_check() != 0)
                return -1;
            xb = ctx->act_bf16;
        }
        CUBLAS_CHECK(cublasGemmEx(ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N, out_dim, rows, in_dim, &alpha,
                                  w->d, CUDA_R_16BF, in_dim, xb, CUDA_R_16BF, x_stride, &beta, y,
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
    return linear_ex(ctx, w, x, 0, y, rows, in_dim, out_dim, in_dim, 0.0f);
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
    return linear_ex(ctx, w, x, 1, y, rows, in_dim, out_dim, x_stride, beta);
}

static int linear_accum(ffwd_cuda_ctx_t *ctx,
                        const cuda_matrix_t *w,
                        const float *x,
                        float *y,
                        int rows,
                        int in_dim,
                        int out_dim) {
    return linear_ex(ctx, w, x, 0, y, rows, in_dim, out_dim, in_dim, 1.0f);
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
// agree with the data type that GEMM reads. BERT runs attention in F32 (passes
// out_bf16=NULL) even with BF16 weights, so it must pass use_bf16=0; the Qwen
// path runs BF16 attention when its weights are BF16. Keying this off the global
// g_weights_bf16 instead silently mismatched the BERT path (BF16 layout, F32
// GEMM) and corrupted equal-length batched attention.
static int
attn_batched_setup(ffwd_cuda_ctx_t *ctx, int batch, int q_offset, int qkv_dim, int use_bf16) {
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

    int bf16 = use_bf16;
    long long want = (long long)G * per;
    if (want > ctx->attn_scores_elems) {
        cudaFree(ctx->attn_scores);
        ctx->attn_scores = NULL;
        ctx->attn_scores_elems = 0;
        CUDA_CHECK(cudaMalloc((void **)&ctx->attn_scores, (size_t)want * sizeof(float)));
        ctx->attn_scores_elems = want;
    }
    if (bf16 && want > ctx->attn_probs_elems) {
        cudaFree(ctx->attn_probs);
        ctx->attn_probs = NULL;
        ctx->attn_probs_elems = 0;
        CUDA_CHECK(cudaMalloc(&ctx->attn_probs, (size_t)want * sizeof(__nv_bfloat16)));
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
    for (int b = 0; b < batch; b++) {
        // Score/probability rows cycle every G sequences (chunked reuse).
        long long srow = (long long)(b % G) * per;
        size_t tok = (size_t)b * L;
        for (int h = 0; h < H; h++) {
            size_t i = (size_t)b * H + h;
            K[i] = ctx->kexp + tok * q_dim + (size_t)h * hd;
            Q[i] = ctx->qkv + tok * qkv_dim + q_offset + (size_t)h * hd;
            S[i] = ctx->attn_scores + srow + (long long)h * L * L;
            if (bf16) {
                V[i] = vb + tok * q_dim + (size_t)h * hd;
                P[i] = pb + srow + (long long)h * L * L;
                O[i] = ab + tok * q_dim + (size_t)h * hd;
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

// out_bf16 != NULL selects the BF16-weight layout: V and the softmax
// probabilities are stored BF16 and the @V GEMM emits BF16 attention output
// at out_bf16 (stride q_dim), ready for the wo projection without a cast.
static int cuda_attention_gemm(ffwd_cuda_ctx_t *ctx,
                               const int *h_offsets,
                               int batch,
                               int q_offset,
                               int k_offset,
                               int v_offset,
                               int qkv_dim,
                               float scale,
                               __nv_bfloat16 *out_bf16) {
    const ffwd_config_t *c = &ctx->config;
    int q_dim = c->q_dim;
    int hd = c->head_dim;
    int H = c->n_heads;
    int total = h_offsets[batch];
    int causal = c->attention_mode == FFWD_ATTENTION_CAUSAL;
    const float alpha = scale, beta = 0.0f, one = 1.0f;

    int threads = 256;
    size_t exp_count = (size_t)total * q_dim;
    if (out_bf16) {
        attn_expand_kv_kernel<<<(exp_count + threads - 1) / threads, threads, 0, ctx->stream>>>(
            ctx->kexp, (__nv_bfloat16 *)ctx->vexp, ctx->qkv, total, H, c->n_kv_heads, hd, qkv_dim,
            k_offset, v_offset);
    } else {
        attn_expand_kv_kernel<<<(exp_count + threads - 1) / threads, threads, 0, ctx->stream>>>(
            ctx->kexp, ctx->vexp, ctx->qkv, total, H, c->n_kv_heads, hd, qkv_dim, k_offset, v_offset);
    }
    if (launch_check() != 0)
        return -1;

    if (ctx->attn_G >= 2) {
        int G = ctx->attn_G, L = ctx->attn_L;
        const void *const *K = (const void *const *)ctx->attn_ptrs;
        const void *const *Q = K + (size_t)batch * H;
        void *const *S = (void *const *)(Q + (size_t)batch * H);
        const void *const *V = (const void *const *)S + (size_t)batch * H;
        const void *const *P = V + (size_t)batch * H;
        void *const *O = (void *const *)(P + (size_t)batch * H);
        cudaDataType pv = out_bf16 ? CUDA_R_16BF : CUDA_R_32F;
        cublasComputeType_t av_ct = out_bf16 ? CUBLAS_COMPUTE_32F : gemm_compute();
        for (int s = 0; s < batch; s += G) {
            int g = batch - s < G ? batch - s : G;
            int n = g * H;
            size_t off = (size_t)s * H;
            /* scores = scale * Q @ K^T, batched over (seq, head) */
            CUBLAS_CHECK(cublasGemmBatchedEx(ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N, L, L, hd, &alpha,
                                             K + off, CUDA_R_32F, q_dim, Q + off, CUDA_R_32F, qkv_dim,
                                             &beta, S + off, CUDA_R_32F, L, n, gemm_compute(),
                                             CUBLAS_GEMM_DEFAULT));
            if (out_bf16) {
                if (causal)
                    attn_causal_softmax_kernel<<<n * L, 256, 0, ctx->stream>>>(
                        ctx->attn_scores, (__nv_bfloat16 *)ctx->attn_probs, L);
                else
                    attn_softmax_kernel<<<n * L, 256, 0, ctx->stream>>>(
                        ctx->attn_scores, (__nv_bfloat16 *)ctx->attn_probs, L);
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
        if (out_bf16) {
            __nv_bfloat16 *P = (__nv_bfloat16 *)ctx->attn_probs;
            const __nv_bfloat16 *V = (const __nv_bfloat16 *)ctx->vexp + (size_t)start * q_dim;
            __nv_bfloat16 *O = out_bf16 + (size_t)start * q_dim;
            if (causal)
                attn_causal_softmax_kernel<<<H * L, 256, 0, ctx->stream>>>(ctx->attn_scores, P, L);
            else
                attn_softmax_kernel<<<H * L, 256, 0, ctx->stream>>>(ctx->attn_scores, P, L);
            if (launch_check() != 0)
                return -1;
            /* O[h] = P[h] @ V[h]  (row-major C = A @ B) */
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(ctx->blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, L, L,
                                                    &one, V, CUDA_R_16BF, q_dim, hd, P, CUDA_R_16BF,
                                                    L, (long long)L * L, &beta, O, CUDA_R_16BF, q_dim,
                                                    hd, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
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

static int ensure_buffers(ffwd_cuda_ctx_t *ctx, int total, int batch, int max_seq) {
    const ffwd_config_t *c = &ctx->config;
    if (total > ctx->seq_cap) {
        cudaFree(ctx->x);
        cudaFree(ctx->x_norm);
        cudaFree(ctx->qkv);
        cudaFree(ctx->attn_out);
        cudaFree(ctx->ffn_gate_up);
        cudaFree(ctx->token_ids);
        cudaFree(ctx->positions);
        cudaFree(ctx->kexp);
        cudaFree(ctx->vexp);
        cudaFree(ctx->act_bf16);
        ctx->x = ctx->x_norm = ctx->qkv = NULL;
        ctx->attn_out = ctx->ffn_gate_up = NULL;
        ctx->token_ids = ctx->positions = NULL;
        ctx->kexp = ctx->vexp = NULL;
        ctx->act_bf16 = NULL;
        CUDA_CHECK(cudaMalloc((void **)&ctx->x, (size_t)total * c->hidden_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->x_norm, (size_t)total * c->hidden_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->qkv,
                              (size_t)total * (c->q_dim + 2 * c->kv_dim) * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->attn_out, (size_t)total * c->q_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->ffn_gate_up,
                              (size_t)total * 2 * c->intermediate_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->token_ids, (size_t)total * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->positions, (size_t)total * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->kexp, (size_t)total * c->q_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->vexp, (size_t)total * c->q_dim * sizeof(float)));
        if (g_weights_bf16)
            CUDA_CHECK(cudaMalloc(&ctx->act_bf16,
                                  (size_t)total * 2 * c->intermediate_size * sizeof(__nv_bfloat16)));
        ctx->seq_cap = total;
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
        long long score_elems = (long long)c->n_heads * max_seq * max_seq;
        if (score_elems > ctx->attn_scores_elems) {
            cudaFree(ctx->attn_scores);
            ctx->attn_scores = NULL;
            ctx->attn_scores_elems = 0;
            CUDA_CHECK(cudaMalloc((void **)&ctx->attn_scores, (size_t)score_elems * sizeof(float)));
            ctx->attn_scores_elems = score_elems;
        }
        if (g_weights_bf16 && score_elems > ctx->attn_probs_elems) {
            cudaFree(ctx->attn_probs);
            ctx->attn_probs = NULL;
            ctx->attn_probs_elems = 0;
            CUDA_CHECK(cudaMalloc(&ctx->attn_probs, (size_t)score_elems * sizeof(__nv_bfloat16)));
            ctx->attn_probs_elems = score_elems;
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

static int load_layer(cuda_layer_t *dst, const ffwd_layer_t *src, const ffwd_config_t *c) {
    return load_qkv_matrix(&dst->qkv, src, c) ||
           load_matrix(&dst->wo, &src->wo, c->hidden_size, c->q_dim) ||
           (c->qkv_bias && load_qkv_bias(&dst->qkv_bias, src, c)) ||
           (c->qk_norm && load_vector(&dst->q_norm, src->q_norm, c->head_dim)) ||
           (c->qk_norm && load_vector(&dst->k_norm, src->k_norm, c->head_dim)) ||
           load_vector(&dst->input_norm, src->input_norm, c->hidden_size) ||
           load_vector(&dst->post_attn_norm, src->post_attn_norm, c->hidden_size) ||
           load_gate_up_matrix(&dst->gate_up_proj, src, c) ||
           load_matrix(&dst->down_proj, &src->down_proj, c->hidden_size, c->intermediate_size);
}

/* BERT layer: attention reuses the fused qkv matrix and qkv_bias (q_dim ==
 * kv_dim == hidden); the two block LayerNorm weights load into input_norm
 * (post-attention) and post_attn_norm (post-FFN); gate_up_proj holds the single
 * up_proj. The added fields carry the biases the Qwen slots lack. */
static int load_bert_layer(cuda_layer_t *dst, const ffwd_layer_t *src, const ffwd_config_t *c) {
    return load_qkv_matrix(&dst->qkv, src, c) ||
           load_matrix(&dst->wo, &src->wo, c->hidden_size, c->q_dim) ||
           load_qkv_bias(&dst->qkv_bias, src, c) ||
           load_vector(&dst->o_bias, src->o_bias, c->hidden_size) ||
           load_vector(&dst->input_norm, src->input_norm, c->hidden_size) ||
           load_vector(&dst->attn_ln_bias, src->attn_ln_bias, c->hidden_size) ||
           load_vector(&dst->post_attn_norm, src->post_attn_norm, c->hidden_size) ||
           load_vector(&dst->ffn_ln_bias, src->ffn_ln_bias, c->hidden_size) ||
           load_matrix(&dst->gate_up_proj, &src->up_proj, c->intermediate_size, c->hidden_size) ||
           load_vector(&dst->ffn_inter_bias, src->ffn_inter_bias, c->intermediate_size) ||
           load_matrix(&dst->down_proj, &src->down_proj, c->hidden_size, c->intermediate_size) ||
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

    /* Without an explicit --gpu-weight-dtype, store weights in the model
     * file's dtype: BF16 snapshots load as BF16 (bit-exact pass-through),
     * F32 snapshots keep the exact F32 default. */
    if (!g_weights_bf16_set) {
        g_weights_bf16 = cpu->weights.embed_tokens.dtype == DTYPE_BF16;
        if (g_weights_bf16)
            fprintf(stderr, "cuda: BF16 model file; storing weights as BF16 "
                            "(use --gpu-weight-dtype f32 to override)\n");
    }

    if (cudaStreamCreate(&ctx->stream) != cudaSuccess ||
        cublasCreate(&ctx->blas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(ctx->blas, ctx->stream) != CUBLAS_STATUS_SUCCESS) {
        ffwd_cuda_free(ctx);
        return NULL;
    }

    const ffwd_config_t *c = &ctx->config;
    ctx->layers = (cuda_layer_t *)calloc((size_t)c->n_layers, sizeof(*ctx->layers));
    if (!ctx->layers) {
        ffwd_cuda_free(ctx);
        return NULL;
    }
    if (load_matrix(&ctx->embed_tokens, &cpu->weights.embed_tokens, c->vocab_size, c->hidden_size) !=
        0) {
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
                     ? load_bert_layer(&ctx->layers[i], &cpu->weights.layers[i], c)
                     : load_layer(&ctx->layers[i], &cpu->weights.layers[i], c);
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
    cudaFree(ctx->attn_out);
    cudaFree(ctx->ffn_gate_up);
    cudaFree(ctx->pooled_out);
    cudaFree(ctx->rope_cos);
    cudaFree(ctx->rope_sin);
    cudaFree(ctx->token_ids);
    cudaFree(ctx->offsets);
    free(ctx->offsets_host);
    cudaFree(ctx->positions);
    cudaFree(ctx->kexp);
    cudaFree(ctx->vexp);
    cudaFree(ctx->act_bf16);
    cudaFree(ctx->weight_f32);
    cudaFree(ctx->attn_scores);
    cudaFree(ctx->attn_probs);
    cudaFree((void *)ctx->attn_ptrs);
    free((void *)ctx->attn_ptrs_host);
    cudaFree(ctx->span_starts);
    cudaFree(ctx->span_lens);
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
// F32 activations: BERT checkpoints are F32, so the linear/linear_accum F32
// entry points are used throughout (they still handle bf16 weights via an
// internal cast, which keeps a future bf16 BERT correct if not yet optimized).
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

    bert_embed_layer_norm_kernel<<<total, 256, ln_smem, ctx->stream>>>(
        ctx->x, ctx->token_ids, ctx->positions, ctx->embed_tokens.d, ctx->embed_tokens.bf16,
        ctx->position_embeddings, ctx->token_type_embedding, ctx->ffwd_ln_w, ctx->ffwd_ln_b, total,
        hidden, c->vocab_size, eps);
    if (launch_check() != 0)
        return -1;

    // BERT attention runs in F32 (cuda_attention_gemm is called with NULL below),
    // so the batched layout must be F32 too, regardless of weight storage dtype.
    if (attn_batched_setup(ctx, batch, q_offset, qkv_dim, 0) != 0)
        return -1;

    for (int layer = 0; layer < c->n_layers; layer++) {
        cuda_layer_t *l = &ctx->layers[layer];

        // Self-attention (post-norm): QKV read x directly, then +bias.
        if (linear(ctx, &l->qkv, ctx->x, ctx->qkv, total, hidden, qkv_dim) != 0)
            return -1;
        int qkv_count = total * qkv_dim;
        add_row_bias_kernel<<<(qkv_count + 255) / 256, 256, 0, ctx->stream>>>(ctx->qkv, l->qkv_bias,
                                                                              total, qkv_dim);
        if (launch_check() != 0)
            return -1;
        if (cuda_attention_gemm(ctx, ctx->offsets_host, batch, q_offset, k_offset, v_offset, qkv_dim,
                                scale, NULL) != 0)
            return -1;

        // Output dense(+bias), residual into x, then post-attention LayerNorm.
        if (linear_accum(ctx, &l->wo, ctx->attn_out, ctx->x, total, c->q_dim, hidden) != 0)
            return -1;
        layer_norm_bias_kernel<<<total, 256, ln_smem, ctx->stream>>>(
            ctx->x, ctx->x, l->o_bias, l->input_norm, l->attn_ln_bias, total, hidden, eps);
        if (launch_check() != 0)
            return -1;

        // Feed-forward (post-norm): dense(+bias) -> GeLU -> dense(+bias).
        if (linear(ctx, &l->gate_up_proj, ctx->x, ctx->ffn_gate_up, total, hidden, inter) != 0)
            return -1;
        int inter_count = total * inter;
        if (c->ffn_act == FFWD_ACT_GELU_TANH)
            bias_gelu_tanh_kernel<<<(inter_count + threads - 1) / threads, threads, 0, ctx->stream>>>(
                ctx->ffn_gate_up, l->ffn_inter_bias, total, inter);
        else
            bias_gelu_kernel<<<(inter_count + threads - 1) / threads, threads, 0, ctx->stream>>>(
                ctx->ffn_gate_up, l->ffn_inter_bias, total, inter);
        if (launch_check() != 0)
            return -1;
        if (linear_accum(ctx, &l->down_proj, ctx->ffn_gate_up, ctx->x, total, inter, hidden) != 0)
            return -1;
        layer_norm_bias_kernel<<<total, 256, ln_smem, ctx->stream>>>(
            ctx->x, ctx->x, l->ffn_out_bias, l->post_attn_norm, l->ffn_ln_bias, total, hidden, eps);
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
        ctx->x, ctx->token_ids, ctx->embed_tokens.d, ctx->embed_tokens.bf16, total, c->hidden_size,
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
    if (attn_batched_setup(ctx, batch, q_offset, qkv_dim, g_weights_bf16) != 0)
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
                                scale, wbf16 ? act : NULL) != 0)
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
            if (linear_ex(ctx, &l->down_proj, ctx->ffn_gate_up, 0, ctx->x, total,
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
        load_matrix(&ctx->projection, proj, ctx->token_dim, ctx->base->config.hidden_size) != 0) {
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
