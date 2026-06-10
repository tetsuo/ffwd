extern "C" {
#include "embed_cuda.h"
#include "embed_internal.h"
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
    float *d;       // device buffer; holds __nv_bfloat16 elements when bf16
    int rows;
    int cols;
    int bf16;       // 1 if d holds __nv_bfloat16, 0 if F32
} cuda_matrix_t;

typedef struct {
    cuda_matrix_t qkv, wo;
    float *q_norm;
    float *k_norm;
    float *input_norm;
    float *post_attn_norm;
    cuda_matrix_t gate_up_proj, down_proj;
} cuda_layer_t;

struct pplx_cuda_ctx {
    pplx_model_t *cpu;
    pplx_config_t config;
    cudaStream_t stream;
    cublasHandle_t blas;
    cuda_matrix_t embed_tokens;
    cuda_layer_t *layers;
    float *norm;

    int seq_cap;
    int batch_cap;
    int max_seq_cap;
    float *x;
    float *x_norm;
    float *qkv;
    float *attn_out;
    float *ffn_gate_up;
    void *act_bf16;     // BF16 cast of a GEMM activation operand (bf16 weights)
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
    int *span_starts;
    int *span_lens;
    int pooled_rows_cap;
    int span_cap;
};

#define CUDA_CHECK(expr) do {                                      \
    cudaError_t _e = (expr);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "cuda: %s failed: %s\n", #expr,            \
                cudaGetErrorString(_e));                           \
        return -1;                                                 \
    }                                                             \
} while (0)

#define CUBLAS_CHECK(expr) do {                                    \
    cublasStatus_t _s = (expr);                                    \
    if (_s != CUBLAS_STATUS_SUCCESS) {                             \
        fprintf(stderr, "cuda: %s failed: cublas status %d\n",     \
                #expr, (int)_s);                                   \
        return -1;                                                 \
    }                                                             \
} while (0)

static int parse_gemm_compute(const char *mode, cublasComputeType_t *out)
{
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

int pplx_cuda_set_fast_gemm(const char *mode)
{
    cublasComputeType_t ct;
    if (parse_gemm_compute(mode, &ct) != 0)
        return -1;
    g_gemm_compute = ct;
    g_gemm_compute_set = 1;
    return 0;
}

static void cuda_matrix_free(cuda_matrix_t *m)
{
    if (!m) return;
    cudaFree(m->d);
    memset(m, 0, sizeof(*m));
}

static float bf16_to_f32(uint16_t v)
{
    uint32_t u = ((uint32_t)v) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// Round F32 to BF16 (truncate the low 16 bits, round to nearest even). NaNs are
// preserved. BF16 shares F32's 8-bit exponent, so this never overflows.
static uint16_t f32_to_bf16(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    if ((x & 0x7fffffffu) > 0x7f800000u)        // NaN
        return (uint16_t)((x >> 16) | 0x0040u);
    x += 0x7fffu + ((x >> 16) & 1u);            // round to nearest even
    return (uint16_t)(x >> 16);
}

// Store weights as native BF16 on the device when set (halves weight memory and
// feeds BF16 tensor cores); default keeps exact F32. Read at load time.
static int g_weights_bf16 = 0;

int pplx_cuda_set_weights_bf16(int on)
{
    g_weights_bf16 = on ? 1 : 0;
    return 0;
}

static int copy_weight_host_f32(float *dst, const pplx_weight_ref_t *w,
                                size_t count)
{
    if (!dst || !w || !w->data) return -1;
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
static int upload_weight(cuda_matrix_t *m, const float *src, size_t count,
                         int rows, int cols)
{
    cudaError_t e;
    if (g_weights_bf16) {
        uint16_t *tmp16 = (uint16_t *)malloc(count * sizeof(uint16_t));
        if (!tmp16) return -1;
        for (size_t i = 0; i < count; i++)
            tmp16[i] = f32_to_bf16(src[i]);
        e = cudaMalloc((void **)&m->d, count * sizeof(__nv_bfloat16));
        if (e != cudaSuccess) {
            fprintf(stderr, "cudaMalloc weight failed: %s\n", cudaGetErrorString(e));
            free(tmp16);
            return -1;
        }
        e = cudaMemcpy(m->d, tmp16, count * sizeof(__nv_bfloat16),
                       cudaMemcpyHostToDevice);
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

static int load_matrix(cuda_matrix_t *m, const pplx_weight_ref_t *w,
                       int rows, int cols)
{
    if (!m || !w || rows <= 0 || cols <= 0) return -1;
    size_t count = (size_t)rows * (size_t)cols;
    float *tmp = (float *)malloc(count * sizeof(float));
    if (!tmp) return -1;
    if (copy_weight_host_f32(tmp, w, count) != 0) {
        free(tmp);
        return -1;
    }
    int r = upload_weight(m, tmp, count, rows, cols);
    free(tmp);
    return r;
}

static int load_qkv_matrix(cuda_matrix_t *m, const pplx_layer_t *src,
                           const pplx_config_t *c)
{
    if (!m || !src || !c) return -1;
    int rows = c->q_dim + 2 * c->kv_dim;
    int cols = c->hidden_size;
    size_t q_count = (size_t)c->q_dim * cols;
    size_t kv_count = (size_t)c->kv_dim * cols;
    size_t total = q_count + 2 * kv_count;
    float *tmp = (float *)malloc(total * sizeof(float));
    if (!tmp) return -1;
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

static int load_gate_up_matrix(cuda_matrix_t *m, const pplx_layer_t *src,
                               const pplx_config_t *c)
{
    if (!m || !src || !c) return -1;
    int rows = 2 * c->intermediate_size;
    int cols = c->hidden_size;
    size_t proj_count = (size_t)c->intermediate_size * cols;
    size_t total = 2 * proj_count;
    float *tmp = (float *)malloc(total * sizeof(float));
    if (!tmp) return -1;
    if (copy_weight_host_f32(tmp, &src->gate_proj, proj_count) != 0 ||
        copy_weight_host_f32(tmp + proj_count, &src->up_proj, proj_count) != 0) {
        free(tmp);
        return -1;
    }
    int r = upload_weight(m, tmp, total, rows, cols);
    free(tmp);
    return r;
}

static int load_vector(float **out, const float *src, int n)
{
    if (!out || !src || n <= 0) return -1;
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

__global__ static void embed_lookup_kernel(float *x, const int *ids,
                                           const void *emb, int emb_bf16,
                                           int total, int hidden, int vocab)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = total * hidden;
    if (idx >= count) return;
    int tok = idx / hidden;
    int d = idx - tok * hidden;
    int id = ids[tok];
    if (id < 0 || id >= vocab) {
        x[idx] = 0.0f;
        return;
    }
    size_t off = (size_t)id * hidden + d;
    x[idx] = emb_bf16 ? __bfloat162float(((const __nv_bfloat16 *)emb)[off])
                      : ((const float *)emb)[off];
}

// Cast n F32 values to BF16 (used to feed BF16-weight projection GEMMs).
__global__ static void cast_f32_to_bf16_kernel(__nv_bfloat16 *out,
                                               const float *in, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

__device__ static float block_sum(float v)
{
    __shared__ float partial[8];
    __shared__ float total;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;

    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    if (lane == 0) partial[warp] = v;
    __syncthreads();

    if (warp == 0) {
        v = lane < warps ? partial[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffff, v, offset);
        if (lane == 0) total = v;
    }
    __syncthreads();
    return total;
}

__device__ static inline void store_act(float *out, size_t i, float v)
{
    out[i] = v;
}

__device__ static inline void store_act(__nv_bfloat16 *out, size_t i, float v)
{
    out[i] = __float2bfloat16(v);
}

// OUT_T is float, or __nv_bfloat16 when the result feeds a BF16-weight
// projection GEMM directly (same rounding as a separate cast kernel).
template <typename OUT_T>
__global__ static void rms_norm_kernel(OUT_T *out, const float *x,
                                       const float *weight, int rows,
                                       int dim, float eps)
{
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
        store_act(out, (size_t)row * dim + d,
                  x[(size_t)row * dim + d] * inv * weight[d]);
}

__global__ static void rms_norm_rope_head_kernel(float *x, const float *weight,
                                                 const int *positions,
                                                 const float *cosv,
                                                 const float *sinv,
                                                 int rows, int heads,
                                                 int head_dim,
                                                 int row_stride,
                                                 int base_offset,
                                                 float eps)
{
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
    size_t base = (size_t)row * row_stride + base_offset + head * head_dim;
    float v = tid < head_dim ? x[base + tid] : 0.0f;
    float pair_v = tid < head_dim ? x[base + pair_d] : 0.0f;
    float sum = v * v;

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 4 ? partial[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) total = sum;
    }
    __syncthreads();

    if (tid < head_dim) {
        float inv = rsqrtf(total / (float)head_dim + eps);
        int pos = positions[row];
        float c = cosv[(size_t)pos * head_dim + tid];
        float s = sinv[(size_t)pos * head_dim + tid];
        float a = v * inv * weight[tid];
        float b = pair_v * inv * weight[pair_d];
        x[base + tid] = a * c + sign * b * s;
    }
}

// Writes SiLU(gate)*up to out with out_stride. The F32 path writes in place
// over the gate half (out = gate_up, out_stride = 2*intermediate); the BF16
// path writes a packed [rows x intermediate] BF16 matrix for the down GEMM.
template <typename OUT_T>
__global__ static void silu_mul_packed_kernel(OUT_T *out, int out_stride,
                                              const float *gate_up, int rows,
                                              int intermediate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * intermediate;
    if (i < n) {
        int row = i / intermediate;
        int d = i - row * intermediate;
        size_t base = (size_t)row * (2 * intermediate);
        float g = gate_up[base + d];
        float u = gate_up[base + intermediate + d];
        store_act(out, (size_t)row * out_stride + d,
                  (g / (1.0f + expf(-g))) * u);
    }
}

// Expand K and V from the n_kv_heads layout to a contiguous [total x q_dim]
// per-query-head layout (GQA: each kv head repeated n_heads/n_kv_heads times)
// so the attention GEMMs can stride uniformly over query heads.
__global__ static void attn_expand_kv_kernel(
    float *kexp, float *vexp, const float *qkv, int total, int n_heads,
    int n_kv_heads, int head_dim, int qkv_dim, int k_offset, int v_offset)
{
    int q_dim = n_heads * head_dim;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)total * q_dim;
    if (idx >= count) return;
    int t = (int)(idx / q_dim);
    int rem = (int)(idx - (size_t)t * q_dim);
    int h = rem / head_dim;
    int e = rem - h * head_dim;
    int kv = h / (n_heads / n_kv_heads);
    const float *base = qkv + (size_t)t * qkv_dim;
    kexp[idx] = base[k_offset + kv * head_dim + e];
    vexp[idx] = base[v_offset + kv * head_dim + e];
}

// Row softmax over the key dimension for one sequence's score tensor laid out
// as scores[head * L * L + i * L + j]; normalizes across j for each (head, i).
// One block per (head, query) row; scores are pre-scaled by the QK^T GEMM.
__global__ static void attn_softmax_kernel(float *scores, int L)
{
    float *s = scores + (size_t)blockIdx.x * L;
    int tid = threadIdx.x;
    __shared__ float red[256];

    float m = -3.402823466e+38F;
    for (int j = tid; j < L; j += blockDim.x) m = fmaxf(m, s[j]);
    red[tid] = m;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]);
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
        if (tid < o) red[tid] += red[tid + o];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    for (int j = tid; j < L; j += blockDim.x) s[j] *= inv;
}

__global__ static void mean_pool_kernel(float *out, const float *x,
                                        const int *offsets, int batch,
                                        int hidden)
{
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

__global__ static void span_pool_kernel(float *out, const float *x,
                                        const int *starts, const int *lens,
                                        int n_spans, int hidden)
{
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

static int launch_check(void)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "cuda kernel failed: %s\n", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

// Compute type for CUDA GEMMs. Default is exact F32 (== cublasSgemm).
// PPLX_CUDA_GEMM_MODE={bf16,tf32,16f} selects a reduced-precision tensor-core
// matmul that keeps F32 inputs/outputs and F32 accumulation but rounds operands
// internally.
static cublasComputeType_t gemm_compute(void)
{
    static int env_init = 0;
    if (!g_gemm_compute_set && !env_init) {
        const char *e = getenv("PPLX_CUDA_GEMM_MODE");
        if (e && parse_gemm_compute(e, &g_gemm_compute) == 0)
            g_gemm_compute_set = 1;
        env_init = 1;
    }
    return g_gemm_compute;
}

// y = w @ x (+ beta*y). With BF16 weights both GEMM inputs are BF16 and
// accumulation/output stay F32: when x_is_bf16 the activation operand was
// already produced in BF16 by the upstream kernel, otherwise the F32 operand
// is cast into ctx->act_bf16 first. With F32 weights this is the exact path,
// honoring gemm_compute().
static int linear_ex(pplx_cuda_ctx_t *ctx, const cuda_matrix_t *w,
                     const void *x, int x_is_bf16, float *y, int rows,
                     int in_dim, int out_dim, int x_stride, float beta)
{
    const float alpha = 1.0f;
    if (w->bf16) {
        const void *xb = x;
        if (!x_is_bf16) {
            size_t n = (size_t)rows * (size_t)x_stride;
            int threads = 256;
            cast_f32_to_bf16_kernel<<<(n + threads - 1) / threads, threads, 0,
                                      ctx->stream>>>(
                (__nv_bfloat16 *)ctx->act_bf16, (const float *)x, n);
            if (launch_check() != 0) return -1;
            xb = ctx->act_bf16;
        }
        CUBLAS_CHECK(cublasGemmEx(ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                  out_dim, rows, in_dim,
                                  &alpha, w->d, CUDA_R_16BF, in_dim,
                                  xb, CUDA_R_16BF, x_stride,
                                  &beta, y, CUDA_R_32F, out_dim,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        return 0;
    }
    CUBLAS_CHECK(cublasGemmEx(ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N,
                              out_dim, rows, in_dim,
                              &alpha, w->d, CUDA_R_32F, in_dim,
                              x, CUDA_R_32F, x_stride,
                              &beta, y, CUDA_R_32F, out_dim,
                              gemm_compute(), CUBLAS_GEMM_DEFAULT));
    return 0;
}

static int linear(pplx_cuda_ctx_t *ctx, const cuda_matrix_t *w,
                  const float *x, float *y, int rows, int in_dim, int out_dim)
{
    return linear_ex(ctx, w, x, 0, y, rows, in_dim, out_dim, in_dim, 0.0f);
}

// x points at activations already stored as BF16 (e.g. ctx->act_bf16).
static int linear_bf16x(pplx_cuda_ctx_t *ctx, const cuda_matrix_t *w,
                        const void *x, float *y, int rows, int in_dim,
                        int out_dim, int x_stride, float beta)
{
    return linear_ex(ctx, w, x, 1, y, rows, in_dim, out_dim, x_stride, beta);
}

static int linear_accum(pplx_cuda_ctx_t *ctx, const cuda_matrix_t *w,
                        const float *x, float *y,
                        int rows, int in_dim, int out_dim)
{
    return linear_ex(ctx, w, x, 0, y, rows, in_dim, out_dim, in_dim, 1.0f);
}

static int gemm_strided_batched(cublasHandle_t blas, cublasOperation_t transa,
                                cublasOperation_t transb, int m, int n, int k,
                                const float *alpha, const float *A, int lda,
                                long long strideA, const float *B, int ldb,
                                long long strideB, const float *beta,
                                float *C, int ldc, long long strideC,
                                int batch_count)
{
    cublasComputeType_t ct = gemm_compute();
    if (ct == CUBLAS_COMPUTE_32F) {
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            blas, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb,
            strideB, beta, C, ldc, strideC, batch_count));
    } else {
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            blas, transa, transb, m, n, k,
            alpha, A, CUDA_R_32F, lda, strideA,
            B, CUDA_R_32F, ldb, strideB,
            beta, C, CUDA_R_32F, ldc, strideC,
            batch_count, ct, CUBLAS_GEMM_DEFAULT));
    }
    return 0;
}

// Attention for one micro-batch. Expands K/V to the per-query-head layout,
// then for each sequence runs batched-over-heads Q@K^T (scaled) -> row softmax
// -> @V via cuBLAS. Requires ctx->kexp, ctx->vexp ([total x q_dim]) and
// ctx->attn_scores ([n_heads x max_seq x max_seq]) to be allocated. h_offsets
// is the host copy of the packed sequence boundaries.
static int cuda_attention_gemm(pplx_cuda_ctx_t *ctx, const int *h_offsets,
                               int batch, int q_offset, int k_offset,
                               int v_offset, int qkv_dim, float scale)
{
    const pplx_config_t *c = &ctx->config;
    int q_dim = c->q_dim;
    int hd = c->head_dim;
    int H = c->n_heads;
    int total = h_offsets[batch];
    const float alpha = scale, beta = 0.0f, one = 1.0f;

    int threads = 256;
    size_t exp_count = (size_t)total * q_dim;
    attn_expand_kv_kernel<<<(exp_count + threads - 1) / threads, threads,
                            0, ctx->stream>>>(
        ctx->kexp, ctx->vexp, ctx->qkv, total, H, c->n_kv_heads, hd, qkv_dim,
        k_offset, v_offset);
    if (launch_check() != 0) return -1;

    for (int b = 0; b < batch; b++) {
        int start = h_offsets[b];
        int L = h_offsets[b + 1] - start;
        const float *Q = ctx->qkv + (size_t)start * qkv_dim + q_offset;
        const float *K = ctx->kexp + (size_t)start * q_dim;
        const float *V = ctx->vexp + (size_t)start * q_dim;
        float *O = ctx->attn_out + (size_t)start * q_dim;
        /* scores[h] = scale * Q[h] @ K[h]^T  (row-major C = A @ B^T) */
        if (gemm_strided_batched(
                ctx->blas, CUBLAS_OP_T, CUBLAS_OP_N, L, L, hd,
                &alpha, K, q_dim, hd, Q, qkv_dim, hd,
                &beta, ctx->attn_scores, L, (long long)L * L, H) != 0)
            return -1;
        attn_softmax_kernel<<<H * L, 256, 0, ctx->stream>>>(
            ctx->attn_scores, L);
        if (launch_check() != 0) return -1;
        /* O[h] = P[h] @ V[h]  (row-major C = A @ B) */
        if (gemm_strided_batched(
                ctx->blas, CUBLAS_OP_N, CUBLAS_OP_N, hd, L, L,
                &one, V, q_dim, hd, ctx->attn_scores, L, (long long)L * L,
                &beta, O, q_dim, hd, H) != 0)
            return -1;
    }
    return 0;
}

static int ensure_buffers(pplx_cuda_ctx_t *ctx, int total, int batch, int max_seq)
{
    const pplx_config_t *c = &ctx->config;
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
        CUDA_CHECK(cudaMalloc((void **)&ctx->x,        (size_t)total * c->hidden_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->x_norm,   (size_t)total * c->hidden_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->qkv,      (size_t)total * (c->q_dim + 2 * c->kv_dim) * sizeof(float)));
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
        if (!ctx->offsets_host) return -1;
        ctx->batch_cap = batch + 1;
    }
    if (max_seq > ctx->max_seq_cap) {
        cudaFree(ctx->rope_cos);
        cudaFree(ctx->rope_sin);
        cudaFree(ctx->attn_scores);
        ctx->attn_scores = NULL;
        CUDA_CHECK(cudaMalloc((void **)&ctx->attn_scores,
                              (size_t)c->n_heads * max_seq * max_seq * sizeof(float)));
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
                float inv = 1.0f / powf(c->rope_theta,
                                        (float)(2 * d) / (float)c->head_dim);
                float angle = (float)pos * inv;
                float cv = cosf(angle);
                float sv = sinf(angle);
                hcos[(size_t)pos * c->head_dim + d] = cv;
                hcos[(size_t)pos * c->head_dim + half + d] = cv;
                hsin[(size_t)pos * c->head_dim + d] = sv;
                hsin[(size_t)pos * c->head_dim + half + d] = sv;
            }
        }
        CUDA_CHECK(cudaMalloc((void **)&ctx->rope_cos,
                              (size_t)max_seq * c->head_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->rope_sin,
                              (size_t)max_seq * c->head_dim * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(ctx->rope_cos, hcos,
                                   (size_t)max_seq * c->head_dim * sizeof(float),
                                   cudaMemcpyHostToDevice, ctx->stream));
        CUDA_CHECK(cudaMemcpyAsync(ctx->rope_sin, hsin,
                                   (size_t)max_seq * c->head_dim * sizeof(float),
                                   cudaMemcpyHostToDevice, ctx->stream));
        free(hcos);
        free(hsin);
        ctx->max_seq_cap = max_seq;
    }
    return 0;
}

static int ensure_pooled_rows(pplx_cuda_ctx_t *ctx, int rows)
{
    if (rows <= ctx->pooled_rows_cap) return 0;
    const pplx_config_t *c = &ctx->config;
    cudaFree(ctx->pooled_out);
    ctx->pooled_out = NULL;
    CUDA_CHECK(cudaMalloc((void **)&ctx->pooled_out,
                          (size_t)rows * c->hidden_size * sizeof(float)));
    ctx->pooled_rows_cap = rows;
    return 0;
}

static int ensure_span_buffers(pplx_cuda_ctx_t *ctx, int n_spans)
{
    if (n_spans <= ctx->span_cap) return 0;
    cudaFree(ctx->span_starts);
    cudaFree(ctx->span_lens);
    ctx->span_starts = NULL;
    ctx->span_lens = NULL;
    CUDA_CHECK(cudaMalloc((void **)&ctx->span_starts,
                          (size_t)n_spans * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&ctx->span_lens,
                          (size_t)n_spans * sizeof(int)));
    ctx->span_cap = n_spans;
    return 0;
}

static int load_layer(cuda_layer_t *dst, const pplx_layer_t *src,
                      const pplx_config_t *c)
{
    return
        load_qkv_matrix(&dst->qkv, src, c) ||
        load_matrix(&dst->wo, &src->wo, c->hidden_size, c->q_dim) ||
        load_vector(&dst->q_norm, src->q_norm, c->head_dim) ||
        load_vector(&dst->k_norm, src->k_norm, c->head_dim) ||
        load_vector(&dst->input_norm, src->input_norm, c->hidden_size) ||
        load_vector(&dst->post_attn_norm, src->post_attn_norm, c->hidden_size) ||
        load_gate_up_matrix(&dst->gate_up_proj, src, c) ||
        load_matrix(&dst->down_proj, &src->down_proj,
                    c->hidden_size, c->intermediate_size);
}

static void free_layer(cuda_layer_t *l)
{
    cuda_matrix_free(&l->qkv);
    cuda_matrix_free(&l->wo);
    cudaFree(l->q_norm);
    cudaFree(l->k_norm);
    cudaFree(l->input_norm);
    cudaFree(l->post_attn_norm);
    cuda_matrix_free(&l->gate_up_proj);
    cuda_matrix_free(&l->down_proj);
}

pplx_cuda_ctx_t *pplx_cuda_load(const char *model_dir)
{
    pplx_model_t *cpu = pplx_model_load(model_dir);
    if (!cpu) return NULL;

    pplx_cuda_ctx_t *ctx = (pplx_cuda_ctx_t *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        pplx_model_free(cpu);
        return NULL;
    }
    ctx->cpu = cpu;
    ctx->config = cpu->config;

    if (cudaStreamCreate(&ctx->stream) != cudaSuccess ||
        cublasCreate(&ctx->blas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(ctx->blas, ctx->stream) != CUBLAS_STATUS_SUCCESS) {
        pplx_cuda_free(ctx);
        return NULL;
    }

    const pplx_config_t *c = &ctx->config;
    ctx->layers = (cuda_layer_t *)calloc((size_t)c->n_layers, sizeof(*ctx->layers));
    if (!ctx->layers) {
        pplx_cuda_free(ctx);
        return NULL;
    }
    if (load_matrix(&ctx->embed_tokens, &cpu->weights.embed_tokens,
                    c->vocab_size, c->hidden_size) != 0 ||
        load_vector(&ctx->norm, cpu->weights.norm, c->hidden_size) != 0) {
        pplx_cuda_free(ctx);
        return NULL;
    }
    for (int i = 0; i < c->n_layers; i++) {
        if (load_layer(&ctx->layers[i], &cpu->weights.layers[i], c) != 0) {
            fprintf(stderr, "cuda: failed to load layer %d\n", i);
            pplx_cuda_free(ctx);
            return NULL;
        }
    }
    cudaStreamSynchronize(ctx->stream);
    return ctx;
}

void pplx_cuda_free(pplx_cuda_ctx_t *ctx)
{
    if (!ctx) return;
    cuda_matrix_free(&ctx->embed_tokens);
    if (ctx->layers) {
        for (int i = 0; i < ctx->config.n_layers; i++)
            free_layer(&ctx->layers[i]);
        free(ctx->layers);
    }
    cudaFree(ctx->norm);
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
    cudaFree(ctx->attn_scores);
    cudaFree(ctx->span_starts);
    cudaFree(ctx->span_lens);
    if (ctx->blas) cublasDestroy(ctx->blas);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    pplx_model_free(ctx->cpu);
    free(ctx);
}

const pplx_config_t *pplx_cuda_config(const pplx_cuda_ctx_t *ctx)
{
    return ctx ? &ctx->config : NULL;
}

static int cuda_forward_batch(pplx_cuda_ctx_t *ctx, const pplx_input_t *inputs,
                              int batch)
{
    if (!ctx || !inputs || batch <= 0) return -1;
    const pplx_config_t *c = &ctx->config;
    int *h_offsets = (int *)malloc((size_t)(batch + 1) * sizeof(int));
    if (!h_offsets) return -1;
    int total = 0, max_seq = 0;
    for (int b = 0; b < batch; b++) {
        if (!inputs[b].ids || inputs[b].n_tokens <= 0) {
            free(h_offsets);
            return -1;
        }
        h_offsets[b] = total;
        total += inputs[b].n_tokens;
        if (inputs[b].n_tokens > max_seq) max_seq = inputs[b].n_tokens;
    }
    h_offsets[batch] = total;
    int *h_ids = (int *)malloc((size_t)total * sizeof(int));
    int *h_pos = (int *)malloc((size_t)total * sizeof(int));
    if (!h_offsets || !h_ids || !h_pos) {
        free(h_offsets); free(h_ids); free(h_pos);
        return -1;
    }

    for (int b = 0; b < batch; b++) {
        int off = h_offsets[b];
        memcpy(h_ids + off, inputs[b].ids,
               (size_t)inputs[b].n_tokens * sizeof(int));
        for (int i = 0; i < inputs[b].n_tokens; i++)
            h_pos[off + i] = i;
    }
    if (ensure_buffers(ctx, total, batch, max_seq) != 0) {
        free(h_offsets); free(h_ids); free(h_pos);
        return -1;
    }
    memcpy(ctx->offsets_host, h_offsets, (size_t)(batch + 1) * sizeof(int));
    cudaError_t ce = cudaMemcpyAsync(ctx->token_ids, h_ids,
                                     (size_t)total * sizeof(int),
                                     cudaMemcpyHostToDevice, ctx->stream);
    if (ce == cudaSuccess)
        ce = cudaMemcpyAsync(ctx->positions, h_pos,
                             (size_t)total * sizeof(int),
                             cudaMemcpyHostToDevice, ctx->stream);
    if (ce == cudaSuccess)
        ce = cudaMemcpyAsync(ctx->offsets, h_offsets,
                             (size_t)(batch + 1) * sizeof(int),
                             cudaMemcpyHostToDevice, ctx->stream);
    free(h_offsets); free(h_ids); free(h_pos);
    if (ce != cudaSuccess) {
        fprintf(stderr, "cuda input copy failed: %s\n", cudaGetErrorString(ce));
        return -1;
    }

    int threads = 256;
    int blocks_hidden = (total * c->hidden_size + threads - 1) / threads;
    embed_lookup_kernel<<<blocks_hidden, threads, 0, ctx->stream>>>(
        ctx->x, ctx->token_ids, ctx->embed_tokens.d, ctx->embed_tokens.bf16,
        total, c->hidden_size, c->vocab_size);
    if (launch_check() != 0) return -1;

    float scale = 1.0f / sqrtf((float)c->head_dim);
    int q_offset = 0;
    int k_offset = c->q_dim;
    int v_offset = c->q_dim + c->kv_dim;
    int qkv_dim = c->q_dim + 2 * c->kv_dim;
    // With BF16 weights the norm and SiLU kernels store their results as BF16
    // directly into ctx->act_bf16, which the next projection GEMM consumes
    // before any later producer reuses the buffer (single stream). This
    // removes three cast launches per layer; only the attention output still
    // needs a cast because cuBLAS cannot emit a BF16 C from F32 A/B operands.
    __nv_bfloat16 *act = (__nv_bfloat16 *)ctx->act_bf16;
    for (int layer = 0; layer < c->n_layers; layer++) {
        cuda_layer_t *l = &ctx->layers[layer];
        int wbf16 = l->qkv.bf16;
        if (wbf16) {
            rms_norm_kernel<<<total, 256, 0, ctx->stream>>>(
                act, ctx->x, l->input_norm, total,
                c->hidden_size, c->rms_norm_eps);
            if (launch_check() != 0) return -1;
            if (linear_bf16x(ctx, &l->qkv, act, ctx->qkv,
                             total, c->hidden_size, qkv_dim,
                             c->hidden_size, 0.0f) != 0)
                return -1;
        } else {
            rms_norm_kernel<<<total, 256, 0, ctx->stream>>>(
                ctx->x_norm, ctx->x, l->input_norm, total,
                c->hidden_size, c->rms_norm_eps);
            if (launch_check() != 0) return -1;
            if (linear(ctx, &l->qkv, ctx->x_norm, ctx->qkv,
                       total, c->hidden_size, qkv_dim) != 0)
                return -1;
        }

        rms_norm_rope_head_kernel<<<dim3(total, c->n_heads), 128, 0, ctx->stream>>>(
            ctx->qkv, l->q_norm, ctx->positions, ctx->rope_cos, ctx->rope_sin,
            total, c->n_heads, c->head_dim, qkv_dim, q_offset,
            c->rms_norm_eps);
        rms_norm_rope_head_kernel<<<dim3(total, c->n_kv_heads), 128, 0, ctx->stream>>>(
            ctx->qkv, l->k_norm, ctx->positions, ctx->rope_cos, ctx->rope_sin,
            total, c->n_kv_heads, c->head_dim, qkv_dim, k_offset,
            c->rms_norm_eps);
        if (launch_check() != 0) return -1;

        if (cuda_attention_gemm(ctx, ctx->offsets_host, batch, q_offset,
                                k_offset, v_offset, qkv_dim, scale) != 0)
            return -1;

        if (linear_accum(ctx, &l->wo, ctx->attn_out, ctx->x,
                         total, c->q_dim, c->hidden_size) != 0)
            return -1;

        int inter_count = total * c->intermediate_size;
        if (wbf16) {
            rms_norm_kernel<<<total, 256, 0, ctx->stream>>>(
                act, ctx->x, l->post_attn_norm, total,
                c->hidden_size, c->rms_norm_eps);
            if (launch_check() != 0) return -1;
            if (linear_bf16x(ctx, &l->gate_up_proj, act, ctx->ffn_gate_up,
                             total, c->hidden_size, 2 * c->intermediate_size,
                             c->hidden_size, 0.0f) != 0)
                return -1;
            silu_mul_packed_kernel<<<(inter_count + threads - 1) / threads,
                                     threads, 0, ctx->stream>>>(
                act, c->intermediate_size, ctx->ffn_gate_up, total,
                c->intermediate_size);
            if (launch_check() != 0) return -1;
            if (linear_bf16x(ctx, &l->down_proj, act, ctx->x,
                             total, c->intermediate_size, c->hidden_size,
                             c->intermediate_size, 1.0f) != 0)
                return -1;
        } else {
            rms_norm_kernel<<<total, 256, 0, ctx->stream>>>(
                ctx->x_norm, ctx->x, l->post_attn_norm, total,
                c->hidden_size, c->rms_norm_eps);
            if (launch_check() != 0) return -1;
            if (linear(ctx, &l->gate_up_proj, ctx->x_norm,
                       ctx->ffn_gate_up, total, c->hidden_size,
                       2 * c->intermediate_size) != 0)
                return -1;
            silu_mul_packed_kernel<<<(inter_count + threads - 1) / threads,
                                     threads, 0, ctx->stream>>>(
                ctx->ffn_gate_up, 2 * c->intermediate_size, ctx->ffn_gate_up,
                total, c->intermediate_size);
            if (launch_check() != 0) return -1;
            if (linear_ex(ctx, &l->down_proj, ctx->ffn_gate_up, 0, ctx->x,
                          total, c->intermediate_size, c->hidden_size,
                          2 * c->intermediate_size, 1.0f) != 0)
                return -1;
        }
    }

    rms_norm_kernel<<<total, 256, 0, ctx->stream>>>(
        ctx->x, ctx->x, ctx->norm, total, c->hidden_size, c->rms_norm_eps);
    if (launch_check() != 0) return -1;

    return 0;
}

int pplx_cuda_embed_batch(pplx_cuda_ctx_t *ctx, const pplx_input_t *inputs,
                          int batch, float *out_embeddings)
{
    if (!ctx || !inputs || batch <= 0 || !out_embeddings) return -1;
    const pplx_config_t *c = &ctx->config;
    if (cuda_forward_batch(ctx, inputs, batch) != 0)
        return -1;
    if (ensure_pooled_rows(ctx, batch) != 0)
        return -1;
    mean_pool_kernel<<<batch, 256, 0, ctx->stream>>>(
        ctx->pooled_out, ctx->x, ctx->offsets, batch, c->hidden_size);
    if (launch_check() != 0) return -1;
    cudaError_t ce = cudaMemcpyAsync(out_embeddings, ctx->pooled_out,
                                     (size_t)batch * c->hidden_size * sizeof(float),
                                     cudaMemcpyDeviceToHost, ctx->stream);
    if (ce == cudaSuccess)
        ce = cudaStreamSynchronize(ctx->stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "cuda output copy failed: %s\n", cudaGetErrorString(ce));
        return -1;
    }
    return 0;
}

int pplx_cuda_embed_spans_batch(pplx_cuda_ctx_t *ctx,
                                const pplx_context_input_t *inputs,
                                int batch, float *out_embeddings)
{
    if (!ctx || !inputs || batch <= 0 || !out_embeddings) return -1;
    const pplx_config_t *c = &ctx->config;
    int total_spans = 0;
    int total_tokens = 0;
    for (int b = 0; b < batch; b++) {
        const pplx_context_input_t *input = &inputs[b];
        int n_tokens = input->input.n_tokens;
        if (!input->input.ids || n_tokens <= 0 || !input->spans ||
            input->n_spans <= 0 || total_spans > INT_MAX - input->n_spans ||
            total_tokens > INT_MAX - n_tokens)
            return -1;
        for (int s = 0; s < input->n_spans; s++) {
            int start = input->spans[s].start;
            int len = input->spans[s].n_tokens;
            if (start < 0 || len <= 0 || start > n_tokens ||
                len > n_tokens - start)
                return -1;
        }
        total_spans += input->n_spans;
        total_tokens += n_tokens;
    }

    pplx_input_t *packed = (pplx_input_t *)malloc((size_t)batch * sizeof(*packed));
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
        const pplx_context_input_t *input = &inputs[b];
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
    if (ensure_pooled_rows(ctx, total_spans) != 0 ||
        ensure_span_buffers(ctx, total_spans) != 0)
        goto cleanup;
    ce = cudaMemcpyAsync(ctx->span_starts, h_starts,
                         (size_t)total_spans * sizeof(int),
                         cudaMemcpyHostToDevice, ctx->stream);
    if (ce == cudaSuccess)
        ce = cudaMemcpyAsync(ctx->span_lens, h_lens,
                             (size_t)total_spans * sizeof(int),
                             cudaMemcpyHostToDevice, ctx->stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "cuda span copy failed: %s\n", cudaGetErrorString(ce));
        goto cleanup;
    }
    span_pool_kernel<<<total_spans, 256, 0, ctx->stream>>>(
        ctx->pooled_out, ctx->x, ctx->span_starts, ctx->span_lens,
        total_spans, c->hidden_size);
    if (launch_check() != 0)
        goto cleanup;
    ce = cudaMemcpyAsync(out_embeddings, ctx->pooled_out,
                         (size_t)total_spans * c->hidden_size * sizeof(float),
                         cudaMemcpyDeviceToHost, ctx->stream);
    if (ce == cudaSuccess)
        ce = cudaStreamSynchronize(ctx->stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "cuda span output copy failed: %s\n",
                cudaGetErrorString(ce));
        goto cleanup;
    }
    rc = 0;

cleanup:
    free(h_lens);
    free(h_starts);
    free(packed);
    return rc;
}

int pplx_cuda_embed_into(pplx_cuda_ctx_t *ctx, const int *token_ids,
                         int n_tokens, float *out_embedding)
{
    pplx_input_t input = { token_ids, n_tokens };
    return pplx_cuda_embed_batch(ctx, &input, 1, out_embedding);
}

float *pplx_cuda_embed(pplx_cuda_ctx_t *ctx, const int *token_ids,
                       int n_tokens)
{
    if (!ctx) return NULL;
    int hidden = ctx->config.hidden_size;
    float *out = (float *)malloc((size_t)hidden * sizeof(float));
    if (!out) return NULL;
    if (pplx_cuda_embed_into(ctx, token_ids, n_tokens, out) != 0) {
        free(out);
        return NULL;
    }
    return out;
}
