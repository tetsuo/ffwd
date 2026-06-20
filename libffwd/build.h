/* ffwd_build.h - build identity and capability information for the ffwd tools
 *
 * The real values come from the Makefile's VERSION_CFLAGS at compile time; the
 * fallbacks below apply when those flags are absent. Front-end only: the
 * inference library does not use these. */
#ifndef FFWD_BUILD_H
#define FFWD_BUILD_H

#include "ffwd.h" /* ffwd_capability(), declared alongside the rest of the API */

#include <stdio.h>
#include <string.h>

#ifndef FFWD_VERSION
#    define FFWD_VERSION "development"
#endif
#ifndef FFWD_BUILD_COMMIT
#    define FFWD_BUILD_COMMIT "unknown"
#endif
#ifndef FFWD_BUILD_DATE
#    define FFWD_BUILD_DATE "unknown"
#endif
#ifndef FFWD_BUILD_OS
#    define FFWD_BUILD_OS "unknown"
#endif
#ifndef FFWD_BUILD_ARCH
#    define FFWD_BUILD_ARCH "unknown"
#endif

/* Program name with any leading directory removed. */
static inline const char *ffwd_prog_name(const char *argv0) {
    const char *slash = strrchr(argv0, '/');
    return slash ? slash + 1 : argv0;
}

static inline void ffwd_print_version(const char *prog) {
    printf("%s %s (%s/%s)\n", prog, FFWD_VERSION, FFWD_BUILD_OS, FFWD_BUILD_ARCH);
}

static inline void ffwd_print_build_info(const char *prog) {
    ffwd_print_version(prog);
    printf("backend:  %s\n"
           "compiler: %s\n"
           "built:    %s\n"
           "commit:   %s\n",
           ffwd_capability(), __VERSION__, FFWD_BUILD_DATE, FFWD_BUILD_COMMIT);
}

#endif /* FFWD_BUILD_H */
