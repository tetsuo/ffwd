CC          = gcc
# Allow FP reassociation and contraction so reduction loops vectorize, but do
# not assume finite math or approximate libm calls: full -ffast-math produced
# NaN embeddings on GCC/OpenBLAS Linux builds (-ffinite-math-only and
# -fapprox-func are the unsafe parts and stay off).
CFLAGS_BASE = -Wall -Wextra -O3 -march=native \
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

.PHONY: all blas mlx cuda shared test bench-tokenizer debug clean help

all: help

help:
	@echo "pplx-embed - Inference for pplx-embed-v1 and pplx-embed-context-v1 embedding models"
	@echo ""
	@echo "Build targets:"
	@echo "  make blas     Build with BLAS acceleration (CPU)"
	@echo "  make mlx      Build with Apple MLX GPU backend (recommended on Apple Silicon)"
	@echo "  make cuda     Build with CUDA/cuBLAS backend (Linux/NVIDIA)"
	@echo "  make shared   Build the shared library ($(SHARED_LIB), BLAS backend)"
	@echo "  make test     Build and run the C test suite (no model files needed)"
	@echo "  make debug    Debug build with AddressSanitizer"
	@echo "  make clean    Remove build artifacts"
	@echo ""
	@echo "Each backend target builds ./$(TARGET) (CLI), ./$(SERVER_TARGET)"
	@echo "(HTTP API), and the static library $(LIB)."
	@echo ""
	@echo "Usage:"
	@echo "  ./pplx-embed -d /path/to/model-dir \"text1\" \"text2\""
	@echo "  ./pplx-embed-server --model pplx-embed-v1-0.6b=/path/to/model-dir"

# =============================================================================
# BLAS build (Apple Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
ifeq ($(strip $(CJSON_CFLAGS)),)
    CJSON_CFLAGS = -I/opt/homebrew/include -I/usr/local/include
endif
ifeq ($(strip $(CJSON_LDFLAGS)),-lcjson)
    CJSON_LDFLAGS = -L/opt/homebrew/lib -L/usr/local/lib -lcjson
endif
blas: CFLAGS  = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate
else
blas: CFLAGS  = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: LDFLAGS += -lopenblas
endif
blas:
	$(MAKE) clean
	$(MAKE) $(TARGET) $(SERVER_TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

# =============================================================================
# MLX build (Apple Silicon GPU - uses mlx-c pure C API)
# Requires: brew install mlx mlx-c
# =============================================================================
MLX_PREFIX  := $(shell brew --prefix mlx 2>/dev/null)
MLXC_PREFIX := $(shell brew --prefix mlx-c 2>/dev/null)

mlx: SRCS += embed_mlx.c
mlx: OBJS  = $(SRCS:.c=.o) embed_mlx.o
ifeq ($(UNAME_S),Darwin)
mlx: CFLAGS  = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK -DUSE_MLX \
               -I$(MLXC_PREFIX)/include
mlx: LDFLAGS += -framework Accelerate -framework Metal -framework Foundation \
                -L$(MLXC_PREFIX)/lib -lmlxc \
                -L$(MLX_PREFIX)/lib -lmlx \
                -Wl,-rpath,$(MLX_PREFIX)/lib -Wl,-rpath,$(MLXC_PREFIX)/lib
endif
mlx:
	$(MAKE) clean
	$(MAKE) $(TARGET) $(SERVER_TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" EXTRA_OBJS="embed_mlx.o"

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
	$(MAKE) $(TARGET) $(SERVER_TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" EXTRA_OBJS="embed_cuda.o"

embed_cuda.o: embed_cuda.cu embed_cuda.h embed_internal.h embed.h
	$(NVCC) -O3 -std=c++17 -DUSE_BLAS -DUSE_OPENBLAS -DUSE_CUDA -I/usr/include/openblas -I$(CUDA_HOME)/include -x cu -c -o $@ $<

# =============================================================================
# Shared library (BLAS backend; override CFLAGS/LDFLAGS for other backends)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
shared: CFLAGS  = $(CFLAGS_BASE) -fPIC -DUSE_BLAS -DACCELERATE_NEW_LAPACK
shared: LDFLAGS += -framework Accelerate
else
shared: CFLAGS  = $(CFLAGS_BASE) -fPIC -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
shared: LDFLAGS += -lopenblas
endif
shared:
	$(MAKE) clean
	$(MAKE) $(SHARED_LIB) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

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

test:
	$(CC) -Wall -Wextra -O2 -I. -o tests/test_kernels_generic \
	    tests/test_kernels.c $(KERNEL_SRCS) -lm -lpthread
	./tests/test_kernels_generic
	$(CC) -Wall -Wextra -O2 $(TEST_BLAS_CFLAGS) -I. -o tests/test_kernels_blas \
	    tests/test_kernels.c $(KERNEL_SRCS) -lm -lpthread $(TEST_BLAS_LDFLAGS)
	./tests/test_kernels_blas
	$(CC) -Wall -Wextra -O2 -I. -o tests/test_safetensors \
	    tests/test_safetensors.c qwen_safetensors.c
	./tests/test_safetensors
	$(CC) -Wall -Wextra -O2 -I. -o tests/test_bf16_model \
	    tests/test_bf16_model.c embed.c qwen_safetensors.c $(KERNEL_SRCS) \
	    -lm -lpthread
	./tests/test_bf16_model

bench-tokenizer:
	$(CC) -Wall -Wextra -O3 -march=native -I. -o bench/bench_tokenizer \
	    bench/bench_tokenizer.c qwen_tokenizer.c $(KERNEL_SRCS) -lm -lpthread
	@echo "run: bench/bench_tokenizer <model_dir> [runs]"

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
	      tests/test_safetensors tests/test_bf16_model bench/bench_tokenizer
