CC          = gcc
# Allow FP reassociation and contraction so reduction loops vectorize, but do
# not assume finite math or approximate libm calls: full -ffast-math produced
# NaN embeddings on GCC/OpenBLAS Linux builds (-ffinite-math-only and
# -fapprox-func are the unsafe parts and stay off).
# -fvisibility=hidden pairs with PPLX_API in the public headers: a new
# public function without the annotation will be missing from the shared lib.
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -fPIC -fvisibility=hidden \
              -fno-math-errno -ffp-contract=fast -fno-trapping-math \
              -fno-signed-zeros -fassociative-math -freciprocal-math
LDFLAGS     = -lm -lpthread
CJSON_CFLAGS ?= $(shell sh -c 'pkg-config --cflags libcjson 2>/dev/null')
CJSON_LDFLAGS ?= $(shell sh -c 'pkg-config --libs libcjson 2>/dev/null')
ifeq ($(strip $(CJSON_LDFLAGS)),)
    CJSON_LDFLAGS = -lcjson
endif

UNAME_S := $(shell uname -s)

# Source files
SRCS = embed.c \
       embed_distributed.c \
       qwen_kernels.c \
       qwen_kernels_generic.c \
       qwen_kernels_neon.c \
       qwen_kernels_avx.c \
       qwen_tokenizer.c \
       qwen_safetensors.c \
       deps/ae/ae.c \
       deps/ae/anet.c \
       deps/ae/monotonic.c

MLX_SRCS = embed_mlx.c
CUDA_SRCS = embed_cuda.cu

OBJS     = $(SRCS:.c=.o)
MLX_OBJS = $(MLX_SRCS:.c=.o)
TARGET        = pplx-embed
SERVER_TARGET = pplx-embed-server
LIB           = libpplxembed.a

KERNEL_SRCS = qwen_kernels.c qwen_kernels_generic.c \
              qwen_kernels_neon.c qwen_kernels_avx.c

ifeq ($(shell uname -s),Darwin)
SHARED_LIB   = libpplxembed.dylib
SHARED_FLAGS = -dynamiclib
else
SHARED_LIB   = libpplxembed.so
SHARED_FLAGS = -shared
endif

.PHONY: all cpu metal cuda test coverage bench bench-model bench-tokenizer \
        debug clean help

all: help

help:
	@echo "pplx-embed - Inference for pplx-embed-v1 and pplx-embed-context-v1 embedding models"
	@echo ""
	@echo "Build targets:"
	@echo "  make cpu      Build the CPU backend (BLAS: Accelerate on macOS, OpenBLAS on Linux)"
	@echo "  make metal    Build the Apple GPU backend via MLX (recommended on Apple Silicon)"
	@echo "  make cuda     Build with CUDA/cuBLAS backend (Linux/NVIDIA)"
	@echo "  make test     Build and run the C test suite (no model files needed)"
	@echo "  make coverage Test-suite line coverage report (clang/llvm-cov)"
	@echo "  make bench    Kernel microbenchmarks; records bench/results/*.json"
	@echo "  make bench-model MODEL_DIR=...  End-to-end embedding benchmark"
	@echo "                (compare records with scripts/benchstat.py)"
	@echo "  make debug    Debug build with AddressSanitizer"
	@echo "  make clean    Remove build artifacts"
	@echo ""
	@echo "Each backend target builds ./$(TARGET) (CLI), ./$(SERVER_TARGET)"
	@echo "(HTTP API), the static library $(LIB), and the shared library"
	@echo "$(SHARED_LIB) for that backend."
	@echo ""
	@echo "Usage:"
	@echo "  ./pplx-embed -d /path/to/model-dir \"text1\" \"text2\""
	@echo "  ./pplx-embed-server --model pplx-embed-v1-0.6b=/path/to/model-dir"

# =============================================================================
# CPU build (BLAS: Apple Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
ifeq ($(strip $(CJSON_CFLAGS)),)
    CJSON_CFLAGS = -I/opt/homebrew/include -I/usr/local/include
endif
ifeq ($(strip $(CJSON_LDFLAGS)),-lcjson)
    CJSON_LDFLAGS = -L/opt/homebrew/lib -L/usr/local/lib -lcjson
endif
cpu: CFLAGS  = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
cpu: LDFLAGS += -framework Accelerate
else
cpu: CFLAGS  = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
cpu: LDFLAGS += -lopenblas
endif
cpu:
	$(MAKE) clean
	$(MAKE) $(TARGET) $(SERVER_TARGET) $(SHARED_LIB) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

# =============================================================================
# Metal build (Apple Silicon GPU via MLX - uses the mlx-c pure C API)
# Requires: brew install mlx mlx-c
# =============================================================================
MLX_PREFIX  := $(shell brew --prefix mlx 2>/dev/null)
MLXC_PREFIX := $(shell brew --prefix mlx-c 2>/dev/null)

metal: SRCS += embed_mlx.c
metal: OBJS  = $(SRCS:.c=.o) embed_mlx.o
ifeq ($(UNAME_S),Darwin)
metal: CFLAGS  = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK -DUSE_MLX \
               -I$(MLXC_PREFIX)/include
metal: LDFLAGS += -framework Accelerate -framework Metal -framework Foundation \
                -L$(MLXC_PREFIX)/lib -lmlxc \
                -L$(MLX_PREFIX)/lib -lmlx \
                -Wl,-rpath,$(MLX_PREFIX)/lib -Wl,-rpath,$(MLXC_PREFIX)/lib
endif
metal:
	$(MAKE) clean
	$(MAKE) $(TARGET) $(SERVER_TARGET) $(SHARED_LIB) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" EXTRA_OBJS="embed_mlx.o"

# =============================================================================
# CUDA build (Linux NVIDIA GPU - uses CUDA Runtime + cuBLAS)
# =============================================================================
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc

cuda: CFLAGS  = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -DUSE_CUDA -I/usr/include/openblas -I$(CUDA_HOME)/include
cuda: LDFLAGS += -L$(CUDA_HOME)/lib64 -lopenblas -lcudart -lcublas
cuda:
	$(MAKE) clean
	$(MAKE) embed_cuda.o
	$(MAKE) $(TARGET) $(SERVER_TARGET) $(SHARED_LIB) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" EXTRA_OBJS="embed_cuda.o"

# -fPIC so the same object links into both binaries and the shared library.
embed_cuda.o: embed_cuda.cu embed_cuda.h embed_internal.h embed.h
	$(NVCC) -O3 -std=c++17 -Xcompiler -fPIC -DUSE_BLAS -DUSE_OPENBLAS -DUSE_CUDA -I/usr/include/openblas -I$(CUDA_HOME)/include -x cu -c -o $@ $<

# =============================================================================
# Shared library (built by every backend target from the same objects)
# =============================================================================
$(SHARED_LIB): $(OBJS) $(EXTRA_OBJS)
	$(CC) $(CFLAGS) $(SHARED_FLAGS) -o $@ $^ $(LDFLAGS) $(CJSON_LDFLAGS)


# =============================================================================
# Tests (no model files required)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
TEST_BLAS_CFLAGS  = -DUSE_BLAS -DACCELERATE_NEW_LAPACK
TEST_BLAS_LDFLAGS = -framework Accelerate
else
TEST_BLAS_CFLAGS  = -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
TEST_BLAS_LDFLAGS = -lopenblas
endif

# Source lists shared by `test` and `coverage` so the two cannot drift.
TEST_CC_FLAGS         = -Wall -Wextra -O2 -I.
TEST_KERNELS_SRCS     = tests/test_kernels.c $(KERNEL_SRCS)
TEST_TOKENIZER_SRCS   = tests/test_tokenizer.c qwen_tokenizer.c $(KERNEL_SRCS)
TEST_SAFETENSORS_SRCS = tests/test_safetensors.c qwen_safetensors.c
TEST_BF16_SRCS        = tests/test_bf16_model.c embed.c qwen_safetensors.c \
                        $(KERNEL_SRCS)
TEST_WORKSPACE_SRCS   = tests/test_workspace.c embed.c qwen_tokenizer.c \
                        qwen_safetensors.c $(KERNEL_SRCS)
TEST_LATE_SRCS        = tests/test_late.c embed.c qwen_tokenizer.c \
                        qwen_safetensors.c $(KERNEL_SRCS)
TEST_SERVER_SRCS      = tests/test_server.c embed.c embed_distributed.c \
                        qwen_tokenizer.c qwen_safetensors.c $(KERNEL_SRCS) \
                        deps/ae/ae.c deps/ae/anet.c deps/ae/monotonic.c
# The CLI check builds the real CLI (CPU backend) plus a driver that runs it.
TEST_CLI_BIN_SRCS     = embed_cli.c embed.c embed_distributed.c \
                        qwen_tokenizer.c qwen_safetensors.c $(KERNEL_SRCS)

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
	$(CC) $(TEST_CC_FLAGS) -o tests/test_safetensors $(TEST_SAFETENSORS_SRCS)
	./tests/test_safetensors
	$(CC) $(TEST_CC_FLAGS) -o tests/test_bf16_model \
	    $(TEST_BF16_SRCS) -lm -lpthread
	./tests/test_bf16_model
	$(CC) $(TEST_CC_FLAGS) -o tests/test_workspace \
	    $(TEST_WORKSPACE_SRCS) -lm -lpthread
	./tests/test_workspace
	$(CC) $(TEST_CC_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/test_late \
	    $(TEST_LATE_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	./tests/test_late
	$(CC) $(TEST_CC_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/cli_under_test \
	    $(TEST_CLI_BIN_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	$(CC) $(TEST_CC_FLAGS) -o tests/test_cli tests/test_cli.c
	./tests/test_cli ./tests/cli_under_test
	$(CC) $(TEST_CC_FLAGS) $(TEST_BLAS_CFLAGS) $(CJSON_CFLAGS) -Ideps/ae \
	    -o tests/test_server $(TEST_SERVER_SRCS) \
	    -lm -lpthread $(TEST_BLAS_LDFLAGS) $(CJSON_LDFLAGS)
	./tests/test_server

# =============================================================================
# Test-suite line coverage (requires clang + llvm-cov/llvm-profdata; this is
# the default toolchain on macOS. Writes a per-file text summary to stdout
# and a browsable HTML report to coverage/html/index.html.)
# =============================================================================
COV_DIR   = coverage
# Source-based coverage; -O0 so no line is folded away by the optimizer.
# The "." compilation dir keeps report paths relative to the repo root
# (run llvm-cov from the repo root so sources resolve).
COV_FLAGS = -fprofile-instr-generate -fcoverage-mapping -O0 \
            -fcoverage-compilation-dir=.
COV_BINS  = tests/test_kernels_generic tests/test_kernels_blas \
            tests/test_tokenizer tests/test_safetensors \
            tests/test_bf16_model tests/test_workspace tests/test_late \
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
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) -o tests/test_safetensors $(TEST_SAFETENSORS_SRCS)
	LLVM_PROFILE_FILE=$(COV_DIR)/safetensors.profraw ./tests/test_safetensors
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) -o tests/test_bf16_model \
	    $(TEST_BF16_SRCS) -lm -lpthread
	LLVM_PROFILE_FILE=$(COV_DIR)/bf16_model.profraw ./tests/test_bf16_model
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) -o tests/test_workspace \
	    $(TEST_WORKSPACE_SRCS) -lm -lpthread
	LLVM_PROFILE_FILE=$(COV_DIR)/workspace.profraw ./tests/test_workspace
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/test_late \
	    $(TEST_LATE_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	LLVM_PROFILE_FILE=$(COV_DIR)/late.profraw ./tests/test_late
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) $(TEST_BLAS_CFLAGS) -o tests/cli_under_test \
	    $(TEST_CLI_BIN_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	$(CC) $(TEST_CC_FLAGS) -o tests/test_cli tests/test_cli.c
	LLVM_PROFILE_FILE=$(COV_DIR)/cli_%p.profraw ./tests/test_cli ./tests/cli_under_test
	$(CC) $(TEST_CC_FLAGS) $(COV_FLAGS) $(TEST_BLAS_CFLAGS) $(CJSON_CFLAGS) -Ideps/ae \
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
	$(CC) -Wall -Wextra -O3 -march=native -I. -o bench/bench_tokenizer \
	    bench/bench_tokenizer.c qwen_tokenizer.c $(KERNEL_SRCS) -lm -lpthread
	@echo "run: bench/bench_tokenizer <model_dir> [runs]"

# =============================================================================
# Regression microbenchmarks (Go-style; see bench/bench.h). Each run records
# a JSON sample set under bench/results/, keyed by commit; compare two
# records with scripts/benchstat.py to see if a change helped or hurt.
# =============================================================================
# Dirtiness ignores .gitignore so local-only ignore entries do not mark
# every record as -dirty.
BENCH_STAMP = $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)$(shell git diff --quiet -- ':!.gitignore' 2>/dev/null || echo -dirty)-$(shell date +%Y%m%d-%H%M%S)
BENCH_CC_FLAGS = -Wall -Wextra $(CFLAGS_BASE) $(TEST_BLAS_CFLAGS) -I.

bench:
	$(CC) $(BENCH_CC_FLAGS) -o bench/bench_kernels \
	    bench/bench_kernels.c $(KERNEL_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	@mkdir -p bench/results
	./bench/bench_kernels --json bench/results/kernels-$(BENCH_STAMP).json
	@echo "compare: scripts/benchstat.py bench/results/OLD.json bench/results/NEW.json"

bench-model:
	@test -n "$(MODEL_DIR)" || { echo "usage: make bench-model MODEL_DIR=/path/to/model-dir"; exit 1; }
	$(CC) $(BENCH_CC_FLAGS) -o bench/bench_model \
	    bench/bench_model.c embed.c qwen_safetensors.c $(KERNEL_SRCS) \
	    -lm -lpthread $(TEST_BLAS_LDFLAGS)
	@mkdir -p bench/results
	./bench/bench_model "$(MODEL_DIR)" --json bench/results/model-$(BENCH_STAMP).json

# =============================================================================
# Debug build
# =============================================================================
SANITIZE ?= address,undefined

debug: CFLAGS  = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=$(SANITIZE)
debug: LDFLAGS += -fsanitize=$(SANITIZE)
debug:
	$(MAKE) clean
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

# =============================================================================
# Static library and binary link
# =============================================================================
$(LIB): $(OBJS) $(EXTRA_OBJS)
	ar rcs $@ $^

$(TARGET): $(LIB) embed_cli.o
	$(CC) $(CFLAGS) -o $@ embed_cli.o $(LIB) $(LDFLAGS) $(CJSON_LDFLAGS)

$(SERVER_TARGET): $(LIB) embed_server.o
	$(CC) $(CFLAGS) -o $@ embed_server.o $(LIB) $(LDFLAGS) $(CJSON_LDFLAGS)

# =============================================================================
# Compile rules
# =============================================================================
embed.o: embed.c embed.h qwen_kernels.h qwen_safetensors.h
	$(CC) $(CFLAGS) $(CJSON_CFLAGS) -Ideps/ae -c -o $@ $<

embed_distributed.o: embed_distributed.c embed_distributed.h embed.h embed_mlx.h
	$(CC) $(CFLAGS) -c -o $@ $<

embed_server.o: embed_server.c embed_server.h embed.h embed_mlx.h qwen_tokenizer.h deps/ae/ae.h deps/ae/anet.h
	$(CC) $(CFLAGS) $(CJSON_CFLAGS) -Ideps/ae -c -o $@ $<

embed_mlx.o: embed_mlx.c embed_mlx.h embed.h qwen_safetensors.h
	$(CC) $(CFLAGS) -c -o $@ $<

embed_cli.o: embed_cli.c embed.h embed_distributed.h qwen_kernels.h qwen_tokenizer.h
	$(CC) $(CFLAGS) $(CJSON_CFLAGS) -Ideps/ae -c -o $@ $<


qwen_kernels.o: qwen_kernels.c qwen_kernels.h qwen_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_kernels_generic.o: qwen_kernels_generic.c qwen_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_kernels_neon.o: qwen_kernels_neon.c qwen_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_kernels_avx.o: qwen_kernels_avx.c qwen_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_tokenizer.o: qwen_tokenizer.c qwen_tokenizer.h qwen_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_safetensors.o: qwen_safetensors.c qwen_safetensors.h
	$(CC) $(CFLAGS) -c -o $@ $<

deps/ae/%.o: deps/ae/%.c deps/ae/ae.h deps/ae/anet.h deps/ae/monotonic.h
	$(CC) $(CFLAGS) -Ideps/ae -c -o $@ $<

# =============================================================================
clean:
	rm -f $(OBJS) embed_mlx.o embed_cuda.o embed_cli.o embed_server.o \
	      $(TARGET) $(SERVER_TARGET) $(LIB) libpplxembed.dylib libpplxembed.so \
	      tests/test_kernels_generic tests/test_kernels_blas \
	      tests/test_safetensors tests/test_bf16_model tests/test_server \
	      tests/test_tokenizer tests/test_workspace \
	      bench/bench_tokenizer
	rm -rf $(COV_DIR)
