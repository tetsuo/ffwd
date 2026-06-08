extern "C" {
#include "pplx_embed_cuda.h"
#include "pplx_embed_internal.h"
}

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *d;
    int rows;
    int cols;
} cuda_matrix_t;

typedef struct {
    cuda_matrix_t wq, wk, wv, wo;
    float *q_norm;
    float *k_norm;
    float *input_norm;
    float *post_attn_norm;
    cuda_matrix_t gate_proj, up_proj, down_proj;
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
    float *q;
    float *k;
    float *v;
    float *attn_out;
    float *proj_out;
    float *ffn_gate;
    float *ffn_up;
    float *rope_cos;
    float *rope_sin;
    int *token_ids;
    int *offsets;
    int *positions;
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
    cudaError_t e = cudaMalloc((void **)&m->d, count * sizeof(float));
    if (e != cudaSuccess) {
        fprintf(stderr, "cudaMalloc weight failed: %s\n", cudaGetErrorString(e));
        free(tmp);
        return -1;
    }
    e = cudaMemcpy(m->d, tmp, count * sizeof(float), cudaMemcpyHostToDevice);
    free(tmp);
    if (e != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy weight failed: %s\n", cudaGetErrorString(e));
        cuda_matrix_free(m);
        return -1;
    }
    m->rows = rows;
    m->cols = cols;
    return 0;
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
                                           const float *emb, int total,
                                           int hidden, int vocab)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = total * hidden;
    if (idx >= count) return;
    int tok = idx / hidden;
    int d = idx - tok * hidden;
    int id = ids[tok];
    x[idx] = (id >= 0 && id < vocab) ? emb[(size_t)id * hidden + d] : 0.0f;
}

__global__ static void rms_norm_kernel(float *out, const float *x,
                                       const float *weight, int rows,
                                       int dim, float eps)
{
    extern __shared__ float sh[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = x[(size_t)row * dim + d];
        sum += v * v;
    }
    sh[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / (float)dim + eps);
    for (int d = tid; d < dim; d += blockDim.x)
        out[(size_t)row * dim + d] = x[(size_t)row * dim + d] * inv * weight[d];
}

__global__ static void rms_norm_head_kernel(float *x, const float *weight,
                                            int rows, int heads, int head_dim,
                                            float eps)
{
    __shared__ float sh[128];
    int row = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    size_t base = ((size_t)row * heads + head) * head_dim;
    float v = tid < head_dim ? x[base + tid] : 0.0f;
    sh[tid] = v * v;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    if (tid < head_dim) {
        float inv = rsqrtf(sh[0] / (float)head_dim + eps);
        x[base + tid] = v * inv * weight[tid];
    }
}

__global__ static void rope_kernel(float *x, const int *positions,
                                   const float *cosv, const float *sinv,
                                   int total, int heads, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int count = total * heads * head_dim;
    if (idx >= count) return;
    int d = idx % head_dim;
    int htmp = idx / head_dim;
    int tok = htmp / heads;
    int half = head_dim / 2;
    int pair_d = d < half ? d + half : d - half;
    float sign = d < half ? -1.0f : 1.0f;
    int pos = positions[tok];
    float c = cosv[(size_t)pos * head_dim + d];
    float s = sinv[(size_t)pos * head_dim + d];
    float a = x[idx];
    float b = x[((size_t)tok * heads + (htmp % heads)) * head_dim + pair_d];
    x[idx] = a * c + sign * b * s;
}

__global__ static void add_kernel(float *x, const float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ static void silu_mul_kernel(float *gate, const float *up, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

__device__ static int find_seq_for_token(const int *offsets, int batch, int tok)
{
    int lo = 0, hi = batch - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (tok < offsets[mid]) hi = mid - 1;
        else if (tok >= offsets[mid + 1]) lo = mid + 1;
        else return mid;
    }
    return 0;
}

__global__ static void gqa_attention_kernel(float *out, const float *q,
                                            const float *k, const float *v,
                                            const int *offsets, int batch,
                                            int total, int n_heads,
                                            int n_kv_heads, int head_dim,
                                            float scale)
{
    __shared__ float red[128];
    __shared__ float score_s;
    __shared__ float max_s;
    __shared__ float sum_s;

    int row = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    int seq = find_seq_for_token(offsets, batch, row);
    int start = offsets[seq];
    int end = offsets[seq + 1];
    int kv_head = head / (n_heads / n_kv_heads);

    const float *qv = q + ((size_t)row * n_heads + head) * head_dim;
    float acc = 0.0f;

    if (tid == 0) max_s = -3.402823466e+38F;
    __syncthreads();
    for (int j = start; j < end; j++) {
        const float *kk = k + ((size_t)j * n_kv_heads + kv_head) * head_dim;
        float part = tid < head_dim ? qv[tid] * kk[tid] : 0.0f;
        red[tid] = part;
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            score_s = red[0] * scale;
            if (score_s > max_s) max_s = score_s;
        }
        __syncthreads();
    }

    if (tid == 0) sum_s = 0.0f;
    __syncthreads();
    for (int j = start; j < end; j++) {
        const float *kk = k + ((size_t)j * n_kv_heads + kv_head) * head_dim;
        float part = tid < head_dim ? qv[tid] * kk[tid] : 0.0f;
        red[tid] = part;
        __syncthreads();
        for (int stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            score_s = expf(red[0] * scale - max_s);
            sum_s += score_s;
        }
        __syncthreads();
        if (tid < head_dim) {
            const float *vv = v + ((size_t)j * n_kv_heads + kv_head) * head_dim;
            acc += score_s * vv[tid];
        }
        __syncthreads();
    }
    if (tid < head_dim)
        out[((size_t)row * n_heads + head) * head_dim + tid] = acc / sum_s;
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

static int launch_check(void)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "cuda kernel failed: %s\n", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

static int linear(cublasHandle_t blas, const cuda_matrix_t *w,
                  const float *x, float *y, int rows, int in_dim, int out_dim)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                             out_dim, rows, in_dim,
                             &alpha, w->d, in_dim, x, in_dim,
                             &beta, y, out_dim));
    return 0;
}

static int ensure_buffers(pplx_cuda_ctx_t *ctx, int total, int batch, int max_seq)
{
    const pplx_config_t *c = &ctx->config;
    if (total > ctx->seq_cap) {
        cudaFree(ctx->x);
        cudaFree(ctx->x_norm);
        cudaFree(ctx->q);
        cudaFree(ctx->k);
        cudaFree(ctx->v);
        cudaFree(ctx->attn_out);
        cudaFree(ctx->proj_out);
        cudaFree(ctx->ffn_gate);
        cudaFree(ctx->ffn_up);
        cudaFree(ctx->token_ids);
        cudaFree(ctx->positions);
        ctx->x = ctx->x_norm = ctx->q = ctx->k = ctx->v = NULL;
        ctx->attn_out = ctx->proj_out = ctx->ffn_gate = ctx->ffn_up = NULL;
        ctx->token_ids = ctx->positions = NULL;
        CUDA_CHECK(cudaMalloc((void **)&ctx->x,        (size_t)total * c->hidden_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->x_norm,   (size_t)total * c->hidden_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->q,        (size_t)total * c->q_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->k,        (size_t)total * c->kv_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->v,        (size_t)total * c->kv_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->attn_out, (size_t)total * c->q_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->proj_out, (size_t)total * c->hidden_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->ffn_gate, (size_t)total * c->intermediate_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->ffn_up,   (size_t)total * c->intermediate_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->token_ids, (size_t)total * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&ctx->positions, (size_t)total * sizeof(int)));
        ctx->seq_cap = total;
    }
    if (batch + 1 > ctx->batch_cap) {
        cudaFree(ctx->offsets);
        CUDA_CHECK(cudaMalloc((void **)&ctx->offsets, (size_t)(batch + 1) * sizeof(int)));
        ctx->batch_cap = batch + 1;
    }
    if (max_seq > ctx->max_seq_cap) {
        cudaFree(ctx->rope_cos);
        cudaFree(ctx->rope_sin);
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

static int load_layer(cuda_layer_t *dst, const pplx_layer_t *src,
                      const pplx_config_t *c)
{
    return
        load_matrix(&dst->wq, &src->wq, c->q_dim, c->hidden_size) ||
        load_matrix(&dst->wk, &src->wk, c->kv_dim, c->hidden_size) ||
        load_matrix(&dst->wv, &src->wv, c->kv_dim, c->hidden_size) ||
        load_matrix(&dst->wo, &src->wo, c->hidden_size, c->q_dim) ||
        load_vector(&dst->q_norm, src->q_norm, c->head_dim) ||
        load_vector(&dst->k_norm, src->k_norm, c->head_dim) ||
        load_vector(&dst->input_norm, src->input_norm, c->hidden_size) ||
        load_vector(&dst->post_attn_norm, src->post_attn_norm, c->hidden_size) ||
        load_matrix(&dst->gate_proj, &src->gate_proj,
                    c->intermediate_size, c->hidden_size) ||
        load_matrix(&dst->up_proj, &src->up_proj,
                    c->intermediate_size, c->hidden_size) ||
        load_matrix(&dst->down_proj, &src->down_proj,
                    c->hidden_size, c->intermediate_size);
}

static void free_layer(cuda_layer_t *l)
{
    cuda_matrix_free(&l->wq);
    cuda_matrix_free(&l->wk);
    cuda_matrix_free(&l->wv);
    cuda_matrix_free(&l->wo);
    cudaFree(l->q_norm);
    cudaFree(l->k_norm);
    cudaFree(l->input_norm);
    cudaFree(l->post_attn_norm);
    cuda_matrix_free(&l->gate_proj);
    cuda_matrix_free(&l->up_proj);
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
    cudaFree(ctx->q);
    cudaFree(ctx->k);
    cudaFree(ctx->v);
    cudaFree(ctx->attn_out);
    cudaFree(ctx->proj_out);
    cudaFree(ctx->ffn_gate);
    cudaFree(ctx->ffn_up);
    cudaFree(ctx->rope_cos);
    cudaFree(ctx->rope_sin);
    cudaFree(ctx->token_ids);
    cudaFree(ctx->offsets);
    cudaFree(ctx->positions);
    if (ctx->blas) cublasDestroy(ctx->blas);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    pplx_model_free(ctx->cpu);
    free(ctx);
}

const pplx_config_t *pplx_cuda_config(const pplx_cuda_ctx_t *ctx)
{
    return ctx ? &ctx->config : NULL;
}

int pplx_cuda_embed_batch(pplx_cuda_ctx_t *ctx, const pplx_input_t *inputs,
                          int batch, float *out_embeddings)
{
    if (!ctx || !inputs || batch <= 0 || !out_embeddings) return -1;
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
        ctx->x, ctx->token_ids, ctx->embed_tokens.d, total,
        c->hidden_size, c->vocab_size);
    if (launch_check() != 0) return -1;

    float scale = 1.0f / sqrtf((float)c->head_dim);
    dim3 q_grid;
    for (int layer = 0; layer < c->n_layers; layer++) {
        cuda_layer_t *l = &ctx->layers[layer];
        rms_norm_kernel<<<total, 256, 256 * sizeof(float), ctx->stream>>>(
            ctx->x_norm, ctx->x, l->input_norm, total,
            c->hidden_size, c->rms_norm_eps);
        if (launch_check() != 0) return -1;

        if (linear(ctx->blas, &l->wq, ctx->x_norm, ctx->q,
                   total, c->hidden_size, c->q_dim) != 0 ||
            linear(ctx->blas, &l->wk, ctx->x_norm, ctx->k,
                   total, c->hidden_size, c->kv_dim) != 0 ||
            linear(ctx->blas, &l->wv, ctx->x_norm, ctx->v,
                   total, c->hidden_size, c->kv_dim) != 0)
            return -1;

        rms_norm_head_kernel<<<dim3(total, c->n_heads), 128, 0, ctx->stream>>>(
            ctx->q, l->q_norm, total, c->n_heads, c->head_dim, c->rms_norm_eps);
        rms_norm_head_kernel<<<dim3(total, c->n_kv_heads), 128, 0, ctx->stream>>>(
            ctx->k, l->k_norm, total, c->n_kv_heads, c->head_dim, c->rms_norm_eps);
        if (launch_check() != 0) return -1;

        q_grid = dim3((total * c->q_dim + threads - 1) / threads);
        rope_kernel<<<q_grid, threads, 0, ctx->stream>>>(
            ctx->q, ctx->positions, ctx->rope_cos, ctx->rope_sin,
            total, c->n_heads, c->head_dim);
        dim3 k_grid((total * c->kv_dim + threads - 1) / threads);
        rope_kernel<<<k_grid, threads, 0, ctx->stream>>>(
            ctx->k, ctx->positions, ctx->rope_cos, ctx->rope_sin,
            total, c->n_kv_heads, c->head_dim);
        if (launch_check() != 0) return -1;

        gqa_attention_kernel<<<dim3(total, c->n_heads), 128, 0, ctx->stream>>>(
            ctx->attn_out, ctx->q, ctx->k, ctx->v, ctx->offsets, batch,
            total, c->n_heads, c->n_kv_heads, c->head_dim, scale);
        if (launch_check() != 0) return -1;

        if (linear(ctx->blas, &l->wo, ctx->attn_out, ctx->proj_out,
                   total, c->q_dim, c->hidden_size) != 0)
            return -1;
        add_kernel<<<blocks_hidden, threads, 0, ctx->stream>>>(
            ctx->x, ctx->proj_out, total * c->hidden_size);
        if (launch_check() != 0) return -1;

        rms_norm_kernel<<<total, 256, 256 * sizeof(float), ctx->stream>>>(
            ctx->x_norm, ctx->x, l->post_attn_norm, total,
            c->hidden_size, c->rms_norm_eps);
        if (launch_check() != 0) return -1;

        if (linear(ctx->blas, &l->gate_proj, ctx->x_norm, ctx->ffn_gate,
                   total, c->hidden_size, c->intermediate_size) != 0 ||
            linear(ctx->blas, &l->up_proj, ctx->x_norm, ctx->ffn_up,
                   total, c->hidden_size, c->intermediate_size) != 0)
            return -1;
        int inter_count = total * c->intermediate_size;
        silu_mul_kernel<<<(inter_count + threads - 1) / threads, threads, 0,
                          ctx->stream>>>(ctx->ffn_gate, ctx->ffn_up, inter_count);
        if (launch_check() != 0) return -1;
        if (linear(ctx->blas, &l->down_proj, ctx->ffn_gate, ctx->proj_out,
                   total, c->intermediate_size, c->hidden_size) != 0)
            return -1;
        add_kernel<<<blocks_hidden, threads, 0, ctx->stream>>>(
            ctx->x, ctx->proj_out, total * c->hidden_size);
        if (launch_check() != 0) return -1;
    }

    rms_norm_kernel<<<total, 256, 256 * sizeof(float), ctx->stream>>>(
        ctx->x, ctx->x, ctx->norm, total, c->hidden_size, c->rms_norm_eps);
    if (launch_check() != 0) return -1;

    float *d_out = NULL;
    ce = cudaMalloc((void **)&d_out, (size_t)batch * c->hidden_size * sizeof(float));
    if (ce != cudaSuccess) {
        fprintf(stderr, "cuda output allocation failed: %s\n",
                cudaGetErrorString(ce));
        return -1;
    }
    mean_pool_kernel<<<batch, 256, 0, ctx->stream>>>(
        d_out, ctx->x, ctx->offsets, batch, c->hidden_size);
    if (launch_check() != 0) {
        cudaFree(d_out);
        return -1;
    }
    ce = cudaMemcpyAsync(out_embeddings, d_out,
                         (size_t)batch * c->hidden_size * sizeof(float),
                         cudaMemcpyDeviceToHost, ctx->stream);
    if (ce == cudaSuccess)
        ce = cudaStreamSynchronize(ctx->stream);
    cudaFree(d_out);
    if (ce != cudaSuccess) {
        fprintf(stderr, "cuda output copy failed: %s\n", cudaGetErrorString(ce));
        return -1;
    }
    return 0;
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
