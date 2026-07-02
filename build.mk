OS   := $(shell uname -s)
ARCH := $(shell uname -m)

CC ?= cc
AR ?= ar
RM ?= rm -f

ARCH_FLAGS ?= -march=native
BACKEND ?= blas
MODE    ?= release
prefix  ?= /usr/local

# Per-configuration build tree under the repo root.
# Each MODE/BACKEND pair gets its own objects and artifacts, so switching configs
# never reuses another config's output and needs no clean.
#
# A consumer sets ROOT, the repo root path, and COMPONENT, a unique name, before
# including this file.
#
# Objects go in $(OBJDIR); final artifacts go in $(OUTDIR).
ROOT      ?= .
COMPONENT ?= misc
BUILDREL  := build/$(MODE)/$(BACKEND)
OUTDIR    := $(ROOT)/$(BUILDREL)
OBJDIR    := $(OUTDIR)/obj/$(COMPONENT)

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDAFLAGS ?= -O3 -std=c++17 -arch=native -Xcompiler -fPIC,-fvisibility=hidden

OPENBLAS_CFLAGS  ?= $(shell sh -c 'pkg-config --cflags openblas 2>/dev/null')
OPENBLAS_LDFLAGS ?= $(shell sh -c 'pkg-config --libs openblas 2>/dev/null')
ifeq ($(strip $(OPENBLAS_CFLAGS)),)
    OPENBLAS_CFLAGS = -I/usr/include/openblas
endif
ifeq ($(strip $(OPENBLAS_LDFLAGS)),)
    OPENBLAS_LDFLAGS = -lopenblas
endif

# yyjson is vendored under deps/yyjson and built into a per-config static archive.
# YYJSON_CFLAGS resolves <yyjson.h> to the vendored header; YYJSON_LDFLAGS names
# the archive by full path (as the server links libae.a), which links the same
# way on macOS and Linux. Consumers build it via $(MAKE) -C deps/yyjson.
YYJSON_CFLAGS  := -I$(ROOT)/deps/yyjson
YYJSON_LDFLAGS := $(OUTDIR)/libyyjson.a

MLX_PREFIX  := $(shell brew --prefix mlx 2>/dev/null)
MLXC_PREFIX := $(shell brew --prefix mlx-c 2>/dev/null)

# Allow FP reassociation and contraction so reduction loops vectorize, without
# assuming finite math or approximate libm calls.
#
# Full -ffast-math produced NaN embeddings on GCC/OpenBLAS Linux builds.
# The unsafe parts, -ffinite-math-only and -fapprox-func, stay off.
#
# -fvisibility=hidden pairs with FFWD_API in public headers. A new public
# function without FFWD_API will be missing from the shared library.
#
# _DEFAULT_SOURCE exposes POSIX/BSD declarations such as strdup and strndup under
# -std=c11. Strict C11 defines __STRICT_ANSI__, so glibc otherwise hides them.
# An implicit int return then truncates the 64-bit pointer and crashes on Linux.
# macOS declares them regardless, which hid this.
#
# -Werror=implicit-function-declaration turns a missing prototype into a build
# failure instead of a buried -Wall warning. That is how the strdup truncation
# above slipped through: implicit int return -> mangled pointer -> Linux-only
# crash. It is also a hard error in C23.
CFLAGS_BASE = -Wall -Wextra -O3 $(ARCH_FLAGS) -fPIC -fvisibility=hidden \
              -D_DEFAULT_SOURCE -Werror=implicit-function-declaration \
              -fno-math-errno -ffp-contract=fast -fno-trapping-math \
              -fno-signed-zeros -fassociative-math -freciprocal-math

SANITIZE ?= address,undefined
ifeq ($(MODE),debug)
    OPT ?= $(CFLAGS_BASE) -pedantic -fno-omit-frame-pointer -g -O0 -DDEBUG -fsanitize=$(SANITIZE)
    LDFLAGS_OPT ?= -fsanitize=$(SANITIZE)
else ifeq ($(MODE),profile)
    OPT ?= $(CFLAGS_BASE) -g -fno-omit-frame-pointer
    LDFLAGS_OPT ?=
else ifeq ($(MODE),release)
    OPT ?= $(CFLAGS_BASE)
    LDFLAGS_OPT ?=
else
    $(error MODE must be release, debug, or profile)
endif

# The shared library exports exactly the public ABI listed in the export map
# (ffwd.exports on Mach-O, ffwd.map on ELF) and hides everything else, so
# the exported set is identical on every backend and no internal symbol leaks.
# This pairs with -fvisibility=hidden. Only the libffwd shared library applies
# EXPORT_FLAGS; the static lib and the binaries link all objects directly.
ifeq ($(OS),Darwin)
    SHARED_EXT   = dylib
    SHARED_FLAGS = -dynamiclib
    EXPORT_FILE  = $(ROOT)/ffwd.exports
    EXPORT_FLAGS = -Wl,-exported_symbols_list,$(ROOT)/ffwd.exports
    HOST_BLAS_CFLAGS  = -DUSE_BLAS -DACCELERATE_NEW_LAPACK
    HOST_BLAS_LDFLAGS = -framework Accelerate
else
    SHARED_EXT   = so
    SHARED_FLAGS = -shared
    EXPORT_FILE  = $(ROOT)/ffwd.map
    EXPORT_FLAGS = -Wl,--version-script,$(ROOT)/ffwd.map
    HOST_BLAS_CFLAGS  = -DUSE_BLAS -DUSE_OPENBLAS $(OPENBLAS_CFLAGS)
    HOST_BLAS_LDFLAGS = $(OPENBLAS_LDFLAGS)
endif

# Each backend defines only the flags needed by its compiled sources.
#
# - blas: CPU build: Kernel files call cblas, so USE_BLAS is enabled.
# - mlx: Apple Metal build: All math runs on the GPU and no kernel files are
#        compiled, so it defines neither USE_BLAS nor links Accelerate.
#        USE_GPU selects MLX through platform.h.
# - cuda: NVIDIA CUDA build:  Still compiles CPU kernel files for host-side work,
#         so it keeps USE_BLAS/OpenBLAS along with the CUDA libraries.
BLAS_BACKEND_CFLAGS  = $(HOST_BLAS_CFLAGS)
BLAS_BACKEND_LDFLAGS = $(HOST_BLAS_LDFLAGS)
MLX_BACKEND_CFLAGS   = -DUSE_GPU -I$(MLXC_PREFIX)/include
MLX_BACKEND_LDFLAGS  = -framework Metal -framework Foundation \
                       -L$(MLXC_PREFIX)/lib -lmlxc \
                       -L$(MLX_PREFIX)/lib -lmlx \
                       -Wl,-rpath,$(MLX_PREFIX)/lib -Wl,-rpath,$(MLXC_PREFIX)/lib
# cuDNN fused attention (libffwd/cuda_sdpa.cu): compiles against the vendored
# cudnn-frontend headers and dlopens libcudnn at runtime, so only cudnn.h is
# needed and only at build time. CUDNN_INC overrides the probe below (point it
# at the directory containing cudnn.h, e.g. a pip nvidia-cudnn-cu12 wheel's
# include/); when no cudnn.h is found the CUDA backend builds without the
# fused path and keeps the built-in kernels.
CUDNN_INC ?=
ifeq ($(strip $(CUDNN_INC)),)
    CUDNN_H := $(firstword $(wildcard /usr/include/cudnn.h \
                 /usr/include/x86_64-linux-gnu/cudnn.h $(CUDA_HOME)/include/cudnn.h))
    ifneq ($(strip $(CUDNN_H)),)
        CUDNN_INC := $(patsubst %/,%,$(dir $(CUDNN_H)))
    endif
endif
ifneq ($(strip $(CUDNN_INC)),)
    CUDNN_SDPA_CFLAGS = -I$(ROOT)/deps/cudnn-frontend/include -I$(CUDNN_INC) \
                        -DFFWD_CUDA_SDPA=1 -DNV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
else
    CUDNN_SDPA_CFLAGS =
endif

CUDA_BACKEND_CFLAGS  = -DUSE_BLAS -DUSE_OPENBLAS -DUSE_GPU $(OPENBLAS_CFLAGS) \
                       -I$(CUDA_HOME)/include $(CUDNN_SDPA_CFLAGS)
# -lstdc++: cuda_sdpa.o uses the C++ runtime (cudnn-frontend), and the tools
# link with the C driver.
CUDA_BACKEND_LDFLAGS = -L$(CUDA_HOME)/lib64 $(OPENBLAS_LDFLAGS) -lcudart -lcublas -lcublasLt \
                       -ldl -lstdc++

ifeq ($(BACKEND),blas)
    BACKEND_CFLAGS  = $(BLAS_BACKEND_CFLAGS)
    BACKEND_LDFLAGS = $(BLAS_BACKEND_LDFLAGS)
else ifeq ($(BACKEND),mlx)
    BACKEND_CFLAGS  = $(MLX_BACKEND_CFLAGS)
    BACKEND_LDFLAGS = $(MLX_BACKEND_LDFLAGS)
else ifeq ($(BACKEND),cuda)
    BACKEND_CFLAGS  = $(CUDA_BACKEND_CFLAGS)
    BACKEND_LDFLAGS = $(CUDA_BACKEND_LDFLAGS)
else
    $(error BACKEND must be blas, mlx, or cuda)
endif

BUILD_DATE   ?= $(shell date -u '+%Y-%m-%dT%H:%M:%SZ')
BUILD_COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
BUILD_OS     ?= $(shell uname -s | tr '[:upper:]' '[:lower:]')
BUILD_ARCH   ?= $(shell uname -m)
VERSION_CFLAGS = \
	-DFFWD_BUILD_DATE=\"$(BUILD_DATE)\" \
	-DFFWD_BUILD_COMMIT=\"$(BUILD_COMMIT)\" \
	-DFFWD_BUILD_OS=\"$(BUILD_OS)\" \
	-DFFWD_BUILD_ARCH=\"$(BUILD_ARCH)\"
ifneq ($(strip $(VERSION)),)
VERSION_CFLAGS += -DFFWD_VERSION=\"$(VERSION)\"
endif

ifdef CFLAGS
else
CFLAGS = $(OPT)
endif
CPPFLAGS ?=
CPPFLAGS += $(YYJSON_CFLAGS)
LDLIBS ?=
SHARED_LDLIBS ?=

objs = $(patsubst %.c,$(OBJDIR)/%.o,$(filter %.c,$1)) $(patsubst %.cu,$(OBJDIR)/%.o,$(filter %.cu,$1))
deps = $(patsubst %.o,%.d,$1)

$(OBJDIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(CPPFLAGS) -MMD -MP -c -o $@ $<

$(OBJDIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(CUDAFLAGS) $(CUDA_BACKEND_CFLAGS) $(CPPFLAGS) -x cu -c -o $@ $<

define static_archive
$(1): $(2) $(3)
	@mkdir -p $$(@D)
	$$(RM) $$@
	$$(AR) rcs $$@ $(2)
endef

define shared_library
$(1): $(2) $(3)
	@mkdir -p $$(@D)
	$$(CC) $$(SHARED_FLAGS) $$(SHARED_LDFLAGS) -o $$@ $(2) $(3) $(4) $$(LDFLAGS_OPT)
endef

define link_program
$(1): $(2) $(3)
	@mkdir -p $$(@D)
	$$(CC) $$(CFLAGS) -o $$@ $(2) $(3) $(4) $$(LDFLAGS_OPT)
endef

define program
OBJS := $$(call objs,$$(SRCS))
DEPS := $$(call deps,$$(OBJS))
.PHONY: all clean
all: $$(BIN)
$$(eval $$(call link_program,$$(BIN),$$(OBJS),$$(LIBS),$$(LDLIBS)))
clean:
	$$(RM) -r $$(OBJS) $$(DEPS) $$(BIN) $$(BIN).dSYM
-include $$(DEPS)
endef

define simple_library
STATIC := $$(OUTDIR)/lib$$(LIB).a
SHARED := $$(OUTDIR)/lib$$(LIB).$$(SHARED_EXT)
OBJS := $$(call objs,$$(SRCS))
DEPS := $$(call deps,$$(OBJS))
.PHONY: all clean install
all: $$(STATIC) $$(SHARED)
$$(eval $$(call static_archive,$$(STATIC),$$(OBJS),))
$$(eval $$(call shared_library,$$(SHARED),$$(OBJS),,$$(SHARED_LDLIBS)))
install: all
	install -d $$(prefix)/lib $$(prefix)/include
	install -m 644 $$(STATIC) $$(prefix)/lib/
	install -m 755 $$(SHARED) $$(prefix)/lib/
	install -m 644 $$(HEADERS) $$(prefix)/include/
clean:
	$$(RM) $$(OBJS) $$(DEPS) $$(STATIC) $$(SHARED)
-include $$(DEPS)
endef

define c_program
$(1): $(2) $(3)
	@mkdir -p $$(@D)
	$$(CC) $$(CFLAGS) $$(CPPFLAGS) $(4) -o $$@ $(2) $(3) $(5) $$(LDFLAGS_OPT)
endef

define c_program_deps
$(1): $(2) $(3)
	@mkdir -p $$(@D)
	$$(CC) $$(CFLAGS) $$(CPPFLAGS) $(5) -o $$@ $(2) $(4) $(6) $$(LDFLAGS_OPT)
endef

export BACKEND MODE ARCH_FLAGS CFLAGS_BASE OPT LDFLAGS_OPT CUDA_HOME NVCC CUDAFLAGS
export OPENBLAS_CFLAGS OPENBLAS_LDFLAGS YYJSON_CFLAGS YYJSON_LDFLAGS
export MLX_PREFIX MLXC_PREFIX
export HOST_BLAS_CFLAGS HOST_BLAS_LDFLAGS
export BLAS_BACKEND_CFLAGS BLAS_BACKEND_LDFLAGS
export MLX_BACKEND_CFLAGS MLX_BACKEND_LDFLAGS
export CUDA_BACKEND_CFLAGS CUDA_BACKEND_LDFLAGS
export BACKEND_CFLAGS BACKEND_LDFLAGS VERSION_CFLAGS
