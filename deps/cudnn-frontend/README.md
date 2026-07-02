# cudnn-frontend (vendored)

Header-only C++ frontend for the cuDNN graph API, used by the CUDA backend for
fused scaled-dot-product attention (`libffwd/cuda_sdpa.cu`).

- Upstream: https://github.com/NVIDIA/cudnn-frontend
- Version: 1.25.0 (commit `91bebc8aef59722c21b39391ea5c16ed3583d959`)
- License: MIT (see `LICENSE.txt`)
- Only `include/` is vendored; samples, docs, and python bindings are dropped.

The frontend is compiled with `NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING`, so
libcudnn is dlopen'd at runtime rather than linked. Building still requires the
cuDNN headers (`cudnn.h`); see the `CUDNN_INC` variable in `build.mk`.

To update: replace `include/` and `LICENSE.txt` from a release tag, then record
the new version and commit here.
