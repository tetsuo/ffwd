/* platform.h - compile-time backend selection.
 *
 * The build passes -DUSE_GPU for a GPU build. This header maps that flag and the
 * host platform to a concrete backend: Apple Silicon uses MLX, any other GPU
 * uses CUDA, and a non-GPU build is CPU. Use USE_GPU for the CPU/GPU split;
 * USE_MLX and USE_CUDA only where a specific accelerator API is called. */
#ifndef FFWD_PLATFORM_H
#define FFWD_PLATFORM_H

#if defined(USE_GPU)
#    if defined(__APPLE__)
#        define USE_MLX 1
#    else
#        define USE_CUDA 1
#    endif
#endif

/* Accelerator label for --build-info and the startup banner; empty on CPU. */
#if defined(USE_MLX)
#    define FFWD_BACKEND_LABEL "Apple Metal"
#elif defined(USE_CUDA)
#    define FFWD_BACKEND_LABEL "NVIDIA CUDA"
#else
#    define FFWD_BACKEND_LABEL ""
#endif

#endif /* FFWD_PLATFORM_H */
