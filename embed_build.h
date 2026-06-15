/* embed_build.h - build identity for the embed and embed-server front-ends.
 *
 * The real values come from the Makefile's VERSION_CFLAGS at compile time; the
 * fallbacks below apply when those flags are absent. Front-end only: the
 * inference library does not use these. */
#ifndef EMBED_BUILD_H
#define EMBED_BUILD_H

#include <stdio.h>
#include <string.h>

#ifndef EMBED_VERSION
#define EMBED_VERSION "development"
#endif
#ifndef EMBED_BUILD_COMMIT
#define EMBED_BUILD_COMMIT "unknown"
#endif
#ifndef EMBED_BUILD_DATE
#define EMBED_BUILD_DATE "unknown"
#endif
#ifndef EMBED_BUILD_OS
#define EMBED_BUILD_OS "unknown"
#endif
#ifndef EMBED_BUILD_ARCH
#define EMBED_BUILD_ARCH "unknown"
#endif

/* Backend label for --build-info, selected by the same macro that picks the
 * compiled inference path. */
#if defined(USE_MLX)
#define EMBED_BUILD_BACKENDS "accelerate, mlx"
#elif defined(USE_CUDA)
#define EMBED_BUILD_BACKENDS "openblas, cuda"
#elif defined(__APPLE__)
#define EMBED_BUILD_BACKENDS "accelerate"
#else
#define EMBED_BUILD_BACKENDS "openblas"
#endif

/* Program name with any leading directory removed. */
static inline const char *embed_prog_name(const char *argv0)
{
    const char *slash = strrchr(argv0, '/');
    return slash ? slash + 1 : argv0;
}

static inline void embed_print_version(const char *prog)
{
    printf("%s %s (%s/%s)\n", prog, EMBED_VERSION,
           EMBED_BUILD_OS, EMBED_BUILD_ARCH);
}

static inline void embed_print_build_info(const char *prog)
{
    embed_print_version(prog);
    printf("\ntarget:   %s-%s [%s]\n"
           "compiler: %s\n"
           "built:    %s\n"
           "commit:   %s\n",
           EMBED_BUILD_OS, EMBED_BUILD_ARCH, EMBED_BUILD_BACKENDS,
           __VERSION__, EMBED_BUILD_DATE, EMBED_BUILD_COMMIT);
}

#endif /* EMBED_BUILD_H */
