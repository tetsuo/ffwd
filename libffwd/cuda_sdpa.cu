/* cuDNN graph-API fused attention (see cuda_sdpa.h for the contract).
 *
 * Compiled against the vendored cudnn-frontend headers with
 * NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING: libcudnn is dlopen'd on first use
 * instead of linked, so the CUDA build carries no hard cuDNN dependency.
 * When the build has no cuDNN headers (FFWD_CUDA_SDPA undefined) this file
 * compiles to stubs that report the feature as unavailable.
 *
 * One graph is built per attention shape class (batch bucket, sequence
 * bucket, head counts, head dim, mask, dtype) and cached; variable lengths
 * inside a bucket are handled by the kernel itself through per-sequence
 * lengths and ragged offsets, so the cache stays small (buckets are powers
 * of two). Shapes the installed cuDNN cannot fuse are negative-cached and
 * the caller falls back to the built-in kernels.
 */

#include "cuda_sdpa.h"

#include <stdio.h>

#ifndef FFWD_CUDA_SDPA

extern "C" ffwd_sdpa_t *ffwd_sdpa_create(void) { return NULL; }
extern "C" void ffwd_sdpa_free(ffwd_sdpa_t *) {}
extern "C" int ffwd_sdpa_run(ffwd_sdpa_t *,
                             cudaStream_t,
                             const void *,
                             const void *,
                             const void *,
                             void *,
                             const int32_t *,
                             const int32_t *,
                             const int32_t *,
                             int,
                             int,
                             int,
                             int,
                             int,
                             float,
                             int,
                             int) {
    return -2;
}

#else /* FFWD_CUDA_SDPA */

#    include <dlfcn.h>

#    include <cstdint>
#    include <cstdlib>
#    include <cstring>
#    include <map>
#    include <memory>
#    include <tuple>
#    include <unordered_map>

#    include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

/* The dynamic-loading shim resolves every cuDNN symbol through this handle;
 * the application owns its definition and initialization. */
namespace cudnn_frontend {
void *cudnn_dlhandle = nullptr;
}

namespace {

/* Variant-pack tensor ids, fixed across all graphs. */
enum : int64_t {
    UID_Q = 1,
    UID_K,
    UID_V,
    UID_O,
    UID_SEQ_Q,
    UID_SEQ_KV,
    UID_RAGGED_Q,
    UID_RAGGED_KV,
    UID_RAGGED_O,
};

struct GraphKey {
    int b, s, hq, hkv, d, causal, f16;
    uint32_t scale_bits;
    bool operator<(const GraphKey &o) const {
        return std::tie(b, s, hq, hkv, d, causal, f16, scale_bits) <
               std::tie(o.b, o.s, o.hq, o.hkv, o.d, o.causal, o.f16, o.scale_bits);
    }
};

/* A built graph, or a negative entry (graph == nullptr) for shapes the
 * installed cuDNN rejected. */
struct GraphEntry {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t workspace_bytes = 0;
};

} // namespace

struct ffwd_sdpa {
    cudnnHandle_t handle = nullptr;
    std::map<GraphKey, GraphEntry> cache;
    void *workspace = nullptr;
    size_t workspace_cap = 0;
    int exec_error_reported = 0;
    int unsupported_reported = 0;
};

/* Builds the fused SDPA graph for one shape class. Returns nullptr when the
 * installed cuDNN does not support it (the caller negative-caches). */
static std::shared_ptr<fe::graph::Graph> sdpa_build_graph(cudnnHandle_t handle, const GraphKey &k) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(k.f16 ? fe::DataType_t::HALF : fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const int64_t b = k.b, hq = k.hq, hkv = k.hkv, d = k.d, s = k.s;

    /* Packed token-major ("THD") tensors: within one sequence the token
     * stride is heads*d and the head stride is d; per-sequence start offsets
     * come from the ragged-offset tensors, which make the nominal batch
     * stride irrelevant for addressing (it still describes the padded
     * bucket). Ragged offsets are element counts, int32, shape (b+1). */
    auto ragged_q = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("ragged_q")
                                      .set_uid(UID_RAGGED_Q)
                                      .set_dim({b + 1, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));
    auto ragged_kv = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("ragged_kv")
                                       .set_uid(UID_RAGGED_KV)
                                       .set_dim({b + 1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
    auto ragged_o = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("ragged_o")
                                      .set_uid(UID_RAGGED_O)
                                      .set_dim({b + 1, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));

    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(UID_Q)
                               .set_dim({b, hq, s, d})
                               .set_stride({s * hq * d, d, hq * d, 1})
                               .set_ragged_offset(ragged_q));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(UID_K)
                               .set_dim({b, hkv, s, d})
                               .set_stride({s * hkv * d, d, hkv * d, 1})
                               .set_ragged_offset(ragged_kv));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(UID_V)
                               .set_dim({b, hkv, s, d})
                               .set_stride({s * hkv * d, d, hkv * d, 1})
                               .set_ragged_offset(ragged_kv));

    auto seq_q = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("seq_q")
                                   .set_uid(UID_SEQ_Q)
                                   .set_dim({b, 1, 1, 1})
                                   .set_stride({1, 1, 1, 1})
                                   .set_data_type(fe::DataType_t::INT32));
    auto seq_kv = graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("seq_kv")
                                    .set_uid(UID_SEQ_KV)
                                    .set_dim({b, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_data_type(fe::DataType_t::INT32));

    float scale;
    std::memcpy(&scale, &k.scale_bits, sizeof(scale));
    auto opts = fe::graph::SDPA_attributes()
                    .set_name("ffwd_sdpa")
                    .set_generate_stats(false)
                    .set_attn_scale(scale)
                    .set_padding_mask(true)
                    .set_seq_len_q(seq_q)
                    .set_seq_len_kv(seq_kv);
    if (k.causal)
        opts.set_causal_mask(true);

    /* Ragged inputs are only expressible by the flash engines (the classic
     * multi-op graph on SM80+, the unified fused op on newer cuDNN), so an
     * unfusable shape fails the build here instead of silently running a
     * materialized fallback. Known gates: GQA+ragged needs cuDNN >= 9.6,
     * ragged on Ampere/Ada (sm8x) needs >= 9.18. */
    auto [O, stats] = graph->sdpa(Q, K, V, opts);
    (void)stats;
    O->set_output(true)
        .set_uid(UID_O)
        .set_dim({b, hq, s, d})
        .set_stride({s * hq * d, d, hq * d, 1})
        .set_ragged_offset(ragged_o);

    if (!graph->build(handle, {fe::HeurMode_t::A}).is_good())
        return nullptr;
    return graph;
}

extern "C" ffwd_sdpa_t *ffwd_sdpa_create(void) {
    const char *lib = getenv("FFWD_CUDNN_LIB");
    void *dl = nullptr;
    if (lib && lib[0])
        dl = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!dl)
        dl = dlopen("libcudnn.so.9", RTLD_NOW | RTLD_LOCAL);
    if (!dl)
        dl = dlopen("libcudnn.so", RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        fprintf(stderr, "ffwd-cuda: libcudnn not found, using built-in attention kernels\n");
        return NULL;
    }
    cudnn_frontend::cudnn_dlhandle = dl;

    ffwd_sdpa_t *s = NULL;
    try {
        /* The fused SDPA support surface used here (ragged THD, GQA head
         * counts, padding mask) is stable from cuDNN 9 on. */
        if (fe::detail::get_backend_version() < 90000) {
            fprintf(stderr, "ffwd-cuda: cuDNN %s too old (need >= 9.0), using built-in kernels\n",
                    fe::detail::get_backend_version_string().c_str());
            return NULL;
        }
        s = new ffwd_sdpa();
        if (fe::detail::create_handle(&s->handle) != CUDNN_STATUS_SUCCESS) {
            delete s;
            return NULL;
        }
    } catch (...) {
        delete s;
        return NULL;
    }
    return s;
}

extern "C" void ffwd_sdpa_free(ffwd_sdpa_t *s) {
    if (!s)
        return;
    s->cache.clear();
    if (s->workspace)
        cudaFree(s->workspace);
    if (s->handle)
        fe::detail::destroy_handle(s->handle);
    delete s;
}

extern "C" int ffwd_sdpa_run(ffwd_sdpa_t *s,
                             cudaStream_t stream,
                             const void *q,
                             const void *k,
                             const void *v,
                             void *o,
                             const int32_t *seq_len,
                             const int32_t *ragged_q,
                             const int32_t *ragged_kv,
                             int batch,
                             int s_max,
                             int hq,
                             int hkv,
                             int d,
                             float scale,
                             int causal,
                             int is_f16) {
    if (!s)
        return -2;
    GraphKey key;
    key.b = batch;
    key.s = s_max;
    key.hq = hq;
    key.hkv = hkv;
    key.d = d;
    key.causal = causal;
    key.f16 = is_f16;
    std::memcpy(&key.scale_bits, &scale, sizeof(scale));

    try {
        auto it = s->cache.find(key);
        if (it == s->cache.end()) {
            GraphEntry e;
            e.graph = sdpa_build_graph(s->handle, key);
            if (e.graph && !e.graph->get_workspace_size(e.workspace_bytes).is_good())
                e.graph = nullptr;
            if (!e.graph && !s->unsupported_reported) {
                s->unsupported_reported = 1;
                fprintf(stderr,
                        "ffwd-cuda: cuDNN %s cannot fuse this attention shape "
                        "(hq=%d hkv=%d d=%d causal=%d), using built-in kernels\n",
                        fe::detail::get_backend_version_string().c_str(), hq, hkv, d, causal);
            }
            it = s->cache.emplace(key, std::move(e)).first;
        }
        if (!it->second.graph)
            return -2;

        if ((size_t)it->second.workspace_bytes > s->workspace_cap) {
            if (s->workspace)
                cudaFree(s->workspace);
            s->workspace = NULL;
            s->workspace_cap = 0;
            if (cudaMalloc(&s->workspace, (size_t)it->second.workspace_bytes) != cudaSuccess)
                return -1;
            s->workspace_cap = (size_t)it->second.workspace_bytes;
        }

        if (fe::detail::set_stream(s->handle, stream) != CUDNN_STATUS_SUCCESS)
            return -1;

        std::unordered_map<int64_t, void *> pack = {
            {UID_Q, const_cast<void *>(q)},
            {UID_K, const_cast<void *>(k)},
            {UID_V, const_cast<void *>(v)},
            {UID_O, o},
            {UID_SEQ_Q, const_cast<int32_t *>(seq_len)},
            {UID_SEQ_KV, const_cast<int32_t *>(seq_len)},
            {UID_RAGGED_Q, const_cast<int32_t *>(ragged_q)},
            {UID_RAGGED_KV, const_cast<int32_t *>(ragged_kv)},
            {UID_RAGGED_O, const_cast<int32_t *>(ragged_q)},
        };
        if (!it->second.graph->execute(s->handle, pack, s->workspace).is_good()) {
            if (!s->exec_error_reported) {
                s->exec_error_reported = 1;
                fprintf(stderr, "ffwd-cuda: cuDNN SDPA execution failed, using built-in kernels\n");
            }
            return -1;
        }
    } catch (const std::exception &ex) {
        if (!s->exec_error_reported) {
            s->exec_error_reported = 1;
            fprintf(stderr, "ffwd-cuda: cuDNN SDPA error: %s\n", ex.what());
        }
        return -1;
    } catch (...) {
        return -1;
    }
    return 0;
}

#endif /* FFWD_CUDA_SDPA */
