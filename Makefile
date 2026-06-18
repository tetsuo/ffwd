CC ?= gcc
AR ?= ar
ARCH_FLAGS ?= -march=native

BUILD_DATE ?= $(shell date -u '+%Y-%m-%dT%H:%M:%SZ')
BUILD_COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
BUILD_OS ?= $(shell uname -s | tr '[:upper:]' '[:lower:]')
BUILD_ARCH ?= $(shell uname -m)

# Version and build identity compiled into the embed and embed-server binaries.
# Applied on the cli.o and server.o recipes. VERSION is set for release builds;
# otherwise the C fallbacks apply.
VERSION_CFLAGS = \
	-DEMBED_BUILD_DATE=\"$(BUILD_DATE)\" \
	-DEMBED_BUILD_COMMIT=\"$(BUILD_COMMIT)\" \
	-DEMBED_BUILD_OS=\"$(BUILD_OS)\" \
	-DEMBED_BUILD_ARCH=\"$(BUILD_ARCH)\"
ifneq ($(strip $(VERSION)),)
VERSION_CFLAGS += -DEMBED_VERSION=\"$(VERSION)\"
endif

# Allow FP reassociation and contraction so reduction loops vectorize, but do
# not assume finite math or approximate libm calls: full -ffast-math produced
# NaN embeddings on GCC/OpenBLAS Linux builds (-ffinite-math-only and
# -fapprox-func are the unsafe parts and stay off).
# -fvisibility=hidden pairs with EMBED_API in the public headers: a new
# public function without the annotation will be missing from the shared lib.
CFLAGS_BASE = -Wall -Wextra -O3 $(ARCH_FLAGS) -fPIC -fvisibility=hidden \
              -fno-math-errno -ffp-contract=fast -fno-trapping-math \
              -fno-signed-zeros -fassociative-math -freciprocal-math
CJSON_CFLAGS ?= $(shell sh -c 'pkg-config --cflags libcjson 2>/dev/null')
CJSON_LDFLAGS ?= $(shell sh -c 'pkg-config --libs libcjson 2>/dev/null')
ifeq ($(strip $(CJSON_LDFLAGS)),)
    CJSON_LDFLAGS = -lcjson
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
ifeq ($(strip $(CJSON_CFLAGS)),)
    CJSON_CFLAGS = -I/opt/homebrew/include -I/usr/local/include
endif
ifeq ($(strip $(CJSON_LDFLAGS)),-lcjson)
    CJSON_LDFLAGS = -L/opt/homebrew/lib -L/usr/local/lib -lcjson
endif
endif

# OpenBLAS lives in different places across distros (/usr/include/openblas on
# Debian/Ubuntu, plain /usr/include on Arch/Alpine, openblas-openmp variants).
# Prefer pkg-config and fall back to the Debian layout so existing setups keep
# working. Used by the Linux/CUDA backends and the test/bench BLAS flags below.
OPENBLAS_CFLAGS  ?= $(shell sh -c 'pkg-config --cflags openblas 2>/dev/null')
OPENBLAS_LDFLAGS ?= $(shell sh -c 'pkg-config --libs openblas 2>/dev/null')
ifeq ($(strip $(OPENBLAS_CFLAGS)),)
    OPENBLAS_CFLAGS = -I/usr/include/openblas
endif
ifeq ($(strip $(OPENBLAS_LDFLAGS)),)
    OPENBLAS_LDFLAGS = -lopenblas
endif

BACKEND ?= cpu
MLX_PREFIX  := $(shell brew --prefix mlx 2>/dev/null)
MLXC_PREFIX := $(shell brew --prefix mlx-c 2>/dev/null)
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc

ifeq ($(BACKEND),mlx)
BACKEND_CFLAGS  = -DUSE_BLAS -DACCELERATE_NEW_LAPACK -DUSE_MLX -I$(MLXC_PREFIX)/include
BACKEND_LDFLAGS = -framework Accelerate -framework Metal -framework Foundation \
                  -L$(MLXC_PREFIX)/lib -lmlxc \
                  -L$(MLX_PREFIX)/lib -lmlx \
                  -Wl,-rpath,$(MLX_PREFIX)/lib -Wl,-rpath,$(MLXC_PREFIX)/lib
EXTRA_OBJS = src/mlx.o
else ifeq ($(BACKEND),cuda)
BACKEND_CFLAGS  = -DUSE_BLAS -DUSE_OPENBLAS -DUSE_CUDA $(OPENBLAS_CFLAGS) -I$(CUDA_HOME)/include
BACKEND_LDFLAGS = -L$(CUDA_HOME)/lib64 $(OPENBLAS_LDFLAGS) -lcudart -lcublas
EXTRA_OBJS = src/cuda.o
else ifeq ($(UNAME_S),Darwin)
BACKEND_CFLAGS  = -DUSE_BLAS -DACCELERATE_NEW_LAPACK
BACKEND_LDFLAGS = -framework Accelerate
else
BACKEND_CFLAGS  = -DUSE_BLAS -DUSE_OPENBLAS $(OPENBLAS_CFLAGS)
BACKEND_LDFLAGS = $(OPENBLAS_LDFLAGS)
endif

# -Isrc so frontends, tests, and benches resolve the public/internal headers
# now that every source lives under src/. EXTRA_CFLAGS/EXTRA_LDFLAGS are the
# append points the debug and prof targets use to layer on sanitizers and
# debug info without dropping the architecture, visibility, or backend flags.
# -MMD -MP make the compiler emit a .d sidecar per object recording the headers
# it included (-MP adds phony targets so a deleted header is not a hard error);
# these are -include'd at the foot of the file so a header edit rebuilds only
# the objects that use it. They live here, not in CFLAGS_BASE, so the one-shot
# test/bench compile+link commands (which reuse CFLAGS_BASE) do not emit stray
# .d files next to their output binaries.
CFLAGS  = $(CFLAGS_BASE) $(BACKEND_CFLAGS) -Isrc -MMD -MP $(EXTRA_CFLAGS)
LDFLAGS = -lm -lpthread $(BACKEND_LDFLAGS) $(EXTRA_LDFLAGS)

# Source groups. CORE_SRCS is the engine the tests and benches reuse; keeping it
# in one variable stops the per-test source lists from drifting apart.
KERNEL_SRCS = src/kernels_gemm.c src/kernels_norm.c src/kernels_act.c src/kernels_attn.c \
              src/kernels_rope.c src/threadpool.c \
              src/kernels_generic.c src/kernels_neon.c src/kernels_avx.c
TOKENIZER_SRCS = src/tokenizer_bpe.c src/tokenizer_wordpiece.c src/tokenizer_sentencepiece.c
CORE_SRCS = src/model.c src/forward.c src/late.c src/vec.c src/alloc.c \
            src/config.c src/safetensors.c $(KERNEL_SRCS)

# Server-only objects: the embed-server binary and the server test link these
# alongside server.o. Not in libembed.a - the CLI and library do not use them.
SERVER_SRCS = src/server_util.c src/sbuf.c src/base64.c src/server_json.c src/server_encode.c src/server_http.c src/server_models.c src/server_handlers.c src/server_schedule.c
SERVER_OBJS = $(SERVER_SRCS:.c=.o)

SRCS = $(CORE_SRCS) $(TOKENIZER_SRCS) \
       deps/ae/ae.c deps/ae/anet.c deps/ae/monotonic.c

MLX_SRCS = src/mlx.c
CUDA_SRCS = src/cuda.cu

OBJS     = $(SRCS:.c=.o)
MLX_OBJS = $(MLX_SRCS:.c=.o)
HEADERS  = $(wildcard src/*.h)
TARGET        = embed
SERVER_TARGET = embed-server
LIB           = libembed.a

ifeq ($(shell uname -s),Darwin)
SHARED_LIB   = libembed.dylib
SHARED_FLAGS = -dynamiclib
else
SHARED_LIB   = libembed.so
SHARED_FLAGS = -shared
endif

.PHONY: all cpu mlx cuda gpu prof test test-bf16 test-safetensors coverage bench \
        bench-model bench-tokenizer bench-server-utils debug clean help force

all: help

help:
	@echo "Targets:"
	@echo "  make cpu      CPU-only build"
	@echo "  make gpu      Build with GPU backend support (mlx on macOS, cuda on Linux)"
	@echo "  make mlx      Apple Silicon GPU via MLX"
	@echo "  make cuda     NVIDIA GPU via CUDA"
	@echo "  make test     Build and run the test suite (no model files needed)"
	@echo "  make coverage Test-suite line coverage report (clang/llvm-cov)"
	@echo "  make bench    Kernel microbenchmarks; records bench/results/*.json"
	@echo "  make bench-model MODEL_DIR=/path/to/model"
	@echo "                End-to-end embedding benchmark"
	@echo "  make bench-server-utils"
	@echo "                Base64 and server string-buffer microbenchmarks"
	@echo "  make debug    Debug build"
	@echo "  make prof     Profiling build"
	@echo "  make clean    Remove build artifacts"

# =============================================================================
# Backend builds: build both binaries and the shared library for one backend.
# =============================================================================
# Switching BACKEND changes CFLAGS but not the source mtimes, so Make alone
# would relink stale objects on a backend switch. The .backend stamp below
# tracks the active backend so these targets keep incremental builds within a
# backend and still rebuild correctly across one.
cpu mlx cuda:
	$(MAKE) BACKEND=$@ $(TARGET) $(SERVER_TARGET) $(SHARED_LIB)

# Stamp recording the backend the current objects were built for. `force` makes
# the recipe run every build, but it rewrites the file (bumping its mtime) only
# when BACKEND actually changed - so the engine objects, which list it as a
# prerequisite, rebuild on a switch and stay untouched otherwise.
.backend: force
	@cur=`cat $@ 2>/dev/null || true`; \
	 [ "$$cur" = "$(BACKEND)" ] || { echo "$(BACKEND)" > $@; }
force:

ifeq ($(UNAME_S),Darwin)
gpu: mlx
else
gpu: cuda
endif

# =============================================================================
# Compile rules
# =============================================================================
# Generic rule for the plain core/kernel/tokenizer/safetensors/mlx objects.
# Sibling headers in src/ resolve via the quote-include search; -Isrc (in
# CFLAGS) makes it explicit. Header prerequisites come from the -MMD-generated
# .d files (-include'd at the foot of the file), which track them per object.
# The .backend prerequisite is attached in bulk further down.
src/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# -fPIC so the same object links into both binaries and the shared library.
src/cuda.o: src/cuda.cu src/cuda.h src/internal.h src/embed.h
	$(NVCC) -O3 -std=c++17 -arch=native -Xcompiler -fPIC,-fvisibility=hidden -DUSE_BLAS -DUSE_OPENBLAS -DUSE_CUDA -Isrc $(OPENBLAS_CFLAGS) -I$(CUDA_HOME)/include -x cu -c -o $@ $<

# server.c reaches deps/ae via "deps/ae/ae.h", so it needs -I. (repo root).
src/server.o: src/server.c
	$(CC) $(CFLAGS) $(VERSION_CFLAGS) $(CJSON_CFLAGS) -I. -Ideps/ae -c -o $@ $<

# Server concern objects: server_internal.h pulls deps/ae/ae.h and cJSON, so
# they need the same -I. and cJSON flags as server.o (the leaf util/sbuf/base64
# objects do not include it but are unharmed by the extra search paths).
$(SERVER_OBJS): src/%.o: src/%.c
	$(CC) $(CFLAGS) $(CJSON_CFLAGS) -I. -Ideps/ae -c -o $@ $<

src/cli.o: src/cli.c
	$(CC) $(CFLAGS) $(VERSION_CFLAGS) -c -o $@ $<

deps/ae/%.o: deps/ae/%.c
	$(CC) $(CFLAGS) -Ideps/ae -c -o $@ $<

# Every engine object is built for one backend; attach the .backend stamp to
# all of them at once (the recipes come from the pattern rules above) so a
# backend switch invalidates them. cli.o/server.o are listed too because they
# inline backend-specific paths through the shared headers.
$(OBJS) $(EXTRA_OBJS) $(SERVER_OBJS) src/cli.o src/server.o: .backend

# =============================================================================
# Shared library (built by every backend target from the same objects)
# =============================================================================
$(SHARED_LIB): $(OBJS) $(EXTRA_OBJS)
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -o $@ $^ $(LDFLAGS)

# =============================================================================
# Static library and binary link
# =============================================================================
$(LIB): $(OBJS) $(EXTRA_OBJS)
	$(AR) rcs $@ $^

$(TARGET): $(LIB) src/cli.o
	$(CC) $(CFLAGS) -o $@ src/cli.o $(LIB) $(LDFLAGS)

$(SERVER_TARGET): $(LIB) src/server.o $(SERVER_OBJS)
	$(CC) $(CFLAGS) -o $@ src/server.o $(SERVER_OBJS) $(LIB) $(LDFLAGS) $(CJSON_LDFLAGS)

# =============================================================================
# Tests (no model files required)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
TEST_BLAS_CFLAGS  = -DUSE_BLAS -DACCELERATE_NEW_LAPACK
TEST_BLAS_LDFLAGS = -framework Accelerate
else
TEST_BLAS_CFLAGS  = -DUSE_BLAS -DUSE_OPENBLAS $(OPENBLAS_CFLAGS)
TEST_BLAS_LDFLAGS = $(OPENBLAS_LDFLAGS)
endif

# Source lists shared by test and coverage targets so the two cannot drift.
TEST_CC_FLAGS         = -Wall -Wextra -O2 -Isrc
TEST_KERNELS_SRCS     = tests/test_kernels.c $(KERNEL_SRCS)
TEST_TOKENIZER_SRCS   = tests/test_tokenizer.c src/tokenizer_bpe.c $(KERNEL_SRCS)
TEST_WORDPIECE_SRCS   = tests/test_wordpiece.c src/tokenizer_wordpiece.c
TEST_SENTENCEPIECE_SRCS = tests/test_sentencepiece.c src/tokenizer_sentencepiece.c
TEST_SAFETENSORS_SRCS = tests/test_safetensors.c src/safetensors.c
TEST_BF16_SRCS        = tests/test_bf16_model.c $(CORE_SRCS)
TEST_QWEN3_SRCS       = tests/test_qwen3.c $(CORE_SRCS)
TEST_QWEN2_SRCS       = tests/test_qwen2.c $(CORE_SRCS)
TEST_CLS_SRCS         = tests/test_cls.c $(CORE_SRCS)
TEST_XLM_ROBERTA_SRCS = tests/test_xlm_roberta.c $(CORE_SRCS)
TEST_WORKSPACE_SRCS   = tests/test_workspace.c $(CORE_SRCS) src/tokenizer_bpe.c
TEST_LATE_SRCS        = tests/test_late.c $(CORE_SRCS) src/tokenizer_bpe.c
# test_server.c #includes ../src/server.c, so server is not a separate object
# here; it needs -I. for server.c's "deps/ae/ae.h" plus the ae sources.
# test_server.c #includes ../src/server.c, so server.c itself is not listed;
# the carved-out server objects (util/sbuf/base64) are separate TUs and must be.
TEST_SERVER_SRCS      = tests/test_server.c $(CORE_SRCS) $(TOKENIZER_SRCS) \
                        $(SERVER_SRCS) \
                        deps/ae/ae.c deps/ae/anet.c deps/ae/monotonic.c
# The CLI check builds the real CLI (CPU backend) plus a driver that runs it.
TEST_CLI_BIN_SRCS     = src/cli.c $(CORE_SRCS) $(TOKENIZER_SRCS)

test:
	$(CC) $(TEST_CC_FLAGS) -o tests/test_kernels_generic \
	    $(TEST_KERNELS_SRCS) -lm -lpthread
	./tests/test_kernels_generic
	$(CC) $(TEST_CC_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/test_kernels_blas \
	    $(TEST_KERNELS_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	./tests/test_kernels_blas
	$(CC) $(TEST_CC_FLAGS) -o tests/test_tokenizer \
	    $(TEST_TOKENIZER_SRCS) -lm -lpthread
	./tests/test_tokenizer
	$(CC) $(TEST_CC_FLAGS) -o tests/test_wordpiece $(TEST_WORDPIECE_SRCS)
	./tests/test_wordpiece
	$(CC) $(TEST_CC_FLAGS) -o tests/test_sentencepiece $(TEST_SENTENCEPIECE_SRCS)
	./tests/test_sentencepiece
	$(CC) $(TEST_CC_FLAGS) -o tests/test_safetensors $(TEST_SAFETENSORS_SRCS)
	./tests/test_safetensors
	$(CC) $(TEST_CC_FLAGS) -o tests/test_bf16_model \
	    $(TEST_BF16_SRCS) -lm -lpthread
	./tests/test_bf16_model
	$(CC) $(TEST_CC_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/test_qwen3 \
	    $(TEST_QWEN3_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	./tests/test_qwen3
	$(CC) $(TEST_CC_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/test_qwen2 \
	    $(TEST_QWEN2_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	./tests/test_qwen2
	$(CC) $(TEST_CC_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/test_cls \
	    $(TEST_CLS_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	./tests/test_cls
	$(CC) $(TEST_CC_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/test_xlm_roberta \
	    $(TEST_XLM_ROBERTA_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	./tests/test_xlm_roberta
	$(CC) $(TEST_CC_FLAGS) -o tests/test_workspace \
	    $(TEST_WORKSPACE_SRCS) -lm -lpthread
	./tests/test_workspace
	$(CC) $(TEST_CC_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/test_late \
	    $(TEST_LATE_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	./tests/test_late
	$(CC) $(TEST_CC_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/cli_under_test \
	    $(TEST_CLI_BIN_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	$(CC) $(TEST_CC_FLAGS) -o tests/test_cli tests/test_cli.c -lm
	./tests/test_cli ./tests/cli_under_test
	$(CC) $(TEST_CC_FLAGS) $(TEST_BLAS_CFLAGS) $(CJSON_CFLAGS) -I. -Ideps/ae \
	    -o tests/test_server $(TEST_SERVER_SRCS) \
	    -lm -lpthread $(TEST_BLAS_LDFLAGS) $(CJSON_LDFLAGS)
	./tests/test_server

test-bf16:
	$(CC) $(TEST_CC_FLAGS) -o tests/test_bf16_model \
	    $(TEST_BF16_SRCS) -lm -lpthread
	./tests/test_bf16_model

test-safetensors:
	$(CC) $(TEST_CC_FLAGS) -o tests/test_safetensors \
	    $(TEST_SAFETENSORS_SRCS) -lm -lpthread
	./tests/test_safetensors

# Test-suite line coverage (requires clang + llvm-cov/llvm-profdata. Writes a
# per-file text summary to stdout and an HTML report to coverage/html/index.html.)
COV_DIR   = coverage
# Source-based coverage; -O0 so no line is folded away by the optimizer.
COV_FLAGS = -fprofile-instr-generate -fcoverage-mapping -O0 \
            -fcoverage-compilation-dir=.
COV_BINS  = tests/test_kernels_generic tests/test_kernels_blas \
            tests/test_tokenizer tests/test_sentencepiece tests/test_safetensors \
            tests/test_bf16_model tests/test_qwen3 tests/test_qwen2 tests/test_workspace \
            tests/test_cls tests/test_xlm_roberta tests/test_late \
            tests/cli_under_test tests/test_server
# Report on project sources only - not the harnesses, vendored deps, or
# system headers (cJSON lands under /opt or /usr otherwise).
COV_IGNORE = -ignore-filename-regex='deps/|tests/|/opt/|/usr/'

ifeq ($(UNAME_S),Darwin)
LLVM_PROFDATA = xcrun llvm-profdata
LLVM_COV      = xcrun llvm-cov
else
LLVM_PROFDATA = llvm-profdata
LLVM_COV      = llvm-cov
endif

coverage:
	rm -rf $(COV_DIR) && mkdir -p $(COV_DIR)
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) -o tests/test_kernels_generic \
	    $(TEST_KERNELS_SRCS) -lm -lpthread
	LLVM_PROFILE_FILE=$(COV_DIR)/kernels_generic.profraw ./tests/test_kernels_generic
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/test_kernels_blas \
	    $(TEST_KERNELS_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	LLVM_PROFILE_FILE=$(COV_DIR)/kernels_blas.profraw ./tests/test_kernels_blas
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) -o tests/test_tokenizer \
	    $(TEST_TOKENIZER_SRCS) -lm -lpthread
	LLVM_PROFILE_FILE=$(COV_DIR)/tokenizer.profraw ./tests/test_tokenizer
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) -o tests/test_sentencepiece \
	    $(TEST_SENTENCEPIECE_SRCS)
	LLVM_PROFILE_FILE=$(COV_DIR)/sentencepiece.profraw ./tests/test_sentencepiece
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) -o tests/test_safetensors $(TEST_SAFETENSORS_SRCS)
	LLVM_PROFILE_FILE=$(COV_DIR)/safetensors.profraw ./tests/test_safetensors
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) -o tests/test_bf16_model \
	    $(TEST_BF16_SRCS) -lm -lpthread
	LLVM_PROFILE_FILE=$(COV_DIR)/bf16_model.profraw ./tests/test_bf16_model
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/test_qwen3 \
	    $(TEST_QWEN3_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	LLVM_PROFILE_FILE=$(COV_DIR)/qwen3.profraw ./tests/test_qwen3
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/test_qwen2 \
	    $(TEST_QWEN2_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	LLVM_PROFILE_FILE=$(COV_DIR)/qwen2.profraw ./tests/test_qwen2
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/test_cls \
	    $(TEST_CLS_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	LLVM_PROFILE_FILE=$(COV_DIR)/cls.profraw ./tests/test_cls
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/test_xlm_roberta \
	    $(TEST_XLM_ROBERTA_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	LLVM_PROFILE_FILE=$(COV_DIR)/xlm_roberta.profraw ./tests/test_xlm_roberta
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) -o tests/test_workspace \
	    $(TEST_WORKSPACE_SRCS) -lm -lpthread
	LLVM_PROFILE_FILE=$(COV_DIR)/workspace.profraw ./tests/test_workspace
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/test_late \
	    $(TEST_LATE_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	LLVM_PROFILE_FILE=$(COV_DIR)/late.profraw ./tests/test_late
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/cli_under_test \
	    $(TEST_CLI_BIN_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	$(CC) $(TEST_CC_FLAGS) -o tests/test_cli tests/test_cli.c -lm
	LLVM_PROFILE_FILE=$(COV_DIR)/cli_%p.profraw ./tests/test_cli ./tests/cli_under_test
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) $(TEST_BLAS_CFLAGS) $(CJSON_CFLAGS) -I. -Ideps/ae \
	    -o tests/test_server $(TEST_SERVER_SRCS) \
	    -lm -lpthread $(TEST_BLAS_LDFLAGS) $(CJSON_LDFLAGS)
	LLVM_PROFILE_FILE=$(COV_DIR)/server.profraw ./tests/test_server
	$(LLVM_PROFDATA) merge -sparse $(COV_DIR)/*.profraw -o $(COV_DIR)/tests.profdata
	$(LLVM_COV) report -instr-profile=$(COV_DIR)/tests.profdata $(COV_IGNORE) \
	    $(addprefix -object ,$(COV_BINS))
	$(LLVM_COV) show \
			-format=html \
			-output-dir=$(COV_DIR)/html \
			-instr-profile=$(COV_DIR)/tests.profdata \
			-show-line-counts-or-regions \
			-show-branches=count \
			-show-expansions \
			-show-instantiations \
			$(COV_IGNORE) \
			$(addprefix -object ,$(COV_BINS))
	@echo "HTML report: $(COV_DIR)/html/index.html"

bench-tokenizer:
	$(CC) -Wall -Wextra -O3 -march=native -Isrc -o bench/bench_tokenizer \
	    bench/bench_tokenizer.c $(TOKENIZER_SRCS) $(KERNEL_SRCS) -lm -lpthread
	@echo "run: bench/bench_tokenizer <model_dir-or-vocab.json> [runs] TEXT..."

# Regression microbenchmarks (see bench/bench.h). Each run records a JSON
# sample set under bench/results/, keyed by commit; compare two records with
# tools/benchstat.py to see if a change helped or hurt.
BENCH_STAMP = $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)$(shell git diff --quiet -- ':!.gitignore' 2>/dev/null || echo -dirty)-$(shell date +%Y%m%d-%H%M%S)
BENCH_CC_FLAGS = -Wall -Wextra $(CFLAGS_BASE) $(TEST_BLAS_CFLAGS) -Isrc
# In-tree alongside the other bench binaries (cleaned by `make clean`) so the
# default works on any OS; the old /private/tmp default existed only on macOS.
BENCH_SERVER_UTILS_BIN ?= bench/bench_server_utils
BENCH_ARGS ?=

bench:
	$(CC) $(BENCH_CC_FLAGS) -o bench/bench_kernels \
	    bench/bench_kernels.c $(KERNEL_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	@mkdir -p bench/results
	./bench/bench_kernels --json bench/results/kernels-$(BENCH_STAMP).json
	@echo "compare: ./tools/benchstat.py bench/results/OLD.json bench/results/NEW.json"

bench-model:
	@test -n "$(MODEL_DIR)" || { echo "usage: make bench-model MODEL_DIR=/path/to/model-dir"; exit 1; }
	$(CC) $(BENCH_CC_FLAGS) -o bench/bench_model \
	    bench/bench_model.c $(CORE_SRCS) \
	    -lm -lpthread $(TEST_BLAS_LDFLAGS)
	@mkdir -p bench/results
	./bench/bench_model "$(MODEL_DIR)" --json bench/results/model-$(BENCH_STAMP).json

bench-server-utils:
	$(CC) $(BENCH_CC_FLAGS) -Ibench -o $(BENCH_SERVER_UTILS_BIN) \
	    bench/bench_server_utils.c src/base64.c src/sbuf.c src/server_util.c \
	    -lm -lpthread $(TEST_BLAS_LDFLAGS)
	$(BENCH_SERVER_UTILS_BIN) $(BENCH_ARGS)

# =============================================================================
# Debug and profiling builds
# =============================================================================
# Both layer extra flags onto the normal build through EXTRA_CFLAGS/EXTRA_LDFLAGS.
# Backend defaults to cpu; override with BACKEND=mlx or BACKEND=cuda.
SANITIZE ?= address,undefined
DEBUG_CFLAGS  = -pedantic -fno-omit-frame-pointer -g -O0 -DDEBUG -fsanitize=$(SANITIZE)
DEBUG_LDFLAGS = -fsanitize=$(SANITIZE)

debug: clean
	$(MAKE) BACKEND=$(BACKEND) \
	    EXTRA_CFLAGS="$(DEBUG_CFLAGS)" EXTRA_LDFLAGS="$(DEBUG_LDFLAGS)" \
	    $(TARGET) $(SERVER_TARGET) $(SHARED_LIB)

prof: clean
	$(MAKE) BACKEND=$(BACKEND) \
	    EXTRA_CFLAGS="-g -fno-omit-frame-pointer" \
	    $(TARGET) $(SERVER_TARGET) $(SHARED_LIB)


# =============================================================================
clean:
	rm -f src/*.o src/*.d deps/ae/*.o deps/ae/*.d .backend \
	      $(TARGET) $(SERVER_TARGET) $(LIB) libembed.dylib libembed.so \
	      tests/test_kernels_generic tests/test_kernels_blas \
	      tests/test_safetensors tests/test_bf16_model tests/test_server \
	      tests/test_qwen3 tests/test_qwen2 tests/test_cls tests/test_xlm_roberta \
	      tests/test_tokenizer tests/test_wordpiece tests/test_sentencepiece \
	      tests/test_workspace tests/test_cli tests/test_late \
	      tests/cli_under_test bench/bench_tokenizer bench/bench_kernels \
	      bench/bench_model bench/bench_server_utils
	rm -rf $(COV_DIR)

# Per-object header dependencies emitted by -MMD -MP. Missing on a clean tree;
# -include ignores them until the first build creates them.
-include $(OBJS:.o=.d) $(EXTRA_OBJS:.o=.d) $(SERVER_OBJS:.o=.d) src/cli.d src/server.d
