CC          = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS     = -lm -lpthread
CJSON_CFLAGS ?= $(shell sh -c 'pkg-config --cflags libcjson 2>/dev/null')
CJSON_LDFLAGS ?= $(shell sh -c 'pkg-config --libs libcjson 2>/dev/null')
ifeq ($(strip $(CJSON_LDFLAGS)),)
    CJSON_LDFLAGS = -lcjson
endif

UNAME_S := $(shell uname -s)

# Source files
SRCS = pplx_embed.c \
       pplx_distributed.c \
       pplx_server.c \
       qwen_asr_kernels.c \
       qwen_asr_kernels_generic.c \
       qwen_asr_kernels_neon.c \
       qwen_asr_kernels_avx.c \
       qwen_asr_tokenizer.c \
       qwen_asr_safetensors.c \
       deps/ae/ae.c \
       deps/ae/anet.c \
       deps/ae/monotonic.c

MLX_SRCS = pplx_embed_mlx.c

OBJS     = $(SRCS:.c=.o)
MLX_OBJS = $(MLX_SRCS:.c=.o)
TARGET   = pplx_embed

.PHONY: all blas mlx debug clean help

all: help

help:
	@echo "pplx_embed - Inference for pplx-embed-v1 and pplx-embed-context-v1 embedding models"
	@echo ""
	@echo "Build targets:"
	@echo "  make blas     Build with BLAS acceleration (CPU)"
	@echo "  make mlx      Build with Apple MLX GPU backend (recommended on Apple Silicon)"
	@echo "  make debug    Debug build with AddressSanitizer"
	@echo "  make clean    Remove build artifacts"
	@echo ""
	@echo "Usage:"
	@echo "  ./pplx_embed -d /path/to/model-dir \"text1\" \"text2\""
	@echo "  ./pplx_embed -d /path/to/model-dir --mlx \"text1\" \"text2\"  (if built with make mlx)"

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
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

# =============================================================================
# MLX build (Apple Silicon GPU - uses mlx-c pure C API)
# Requires: brew install mlx mlx-c
# =============================================================================
MLX_PREFIX  := $(shell brew --prefix mlx 2>/dev/null)
MLXC_PREFIX := $(shell brew --prefix mlx-c 2>/dev/null)

mlx: SRCS += pplx_embed_mlx.c
mlx: OBJS  = $(SRCS:.c=.o) pplx_embed_mlx.o
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
	$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" EXTRA_OBJS="pplx_embed_mlx.o"

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
# Link
# =============================================================================
$(TARGET): $(OBJS) $(EXTRA_OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(CJSON_LDFLAGS)

# =============================================================================
# Compile rules
# =============================================================================
pplx_embed.o: pplx_embed.c pplx_embed.h qwen_asr_kernels.h qwen_asr_safetensors.h
	$(CC) $(CFLAGS) $(CJSON_CFLAGS) -Ideps/ae -c -o $@ $<

pplx_distributed.o: pplx_distributed.c pplx_distributed.h pplx_embed.h pplx_embed_mlx.h
	$(CC) $(CFLAGS) -c -o $@ $<

pplx_server.o: pplx_server.c pplx_server.h pplx_embed.h pplx_embed_mlx.h qwen_asr_tokenizer.h deps/ae/ae.h deps/ae/anet.h
	$(CC) $(CFLAGS) $(CJSON_CFLAGS) -Ideps/ae -c -o $@ $<

pplx_embed_mlx.o: pplx_embed_mlx.c pplx_embed_mlx.h pplx_embed.h qwen_asr_safetensors.h
	$(CC) $(CFLAGS) -c -o $@ $<

main.o: main.c pplx_embed.h pplx_distributed.h pplx_server.h qwen_asr_kernels.h qwen_asr_tokenizer.h
	$(CC) $(CFLAGS) $(CJSON_CFLAGS) -Ideps/ae -c -o $@ $<

qwen_asr_kernels.o: qwen_asr_kernels.c qwen_asr_kernels.h qwen_asr_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_asr_kernels_generic.o: qwen_asr_kernels_generic.c qwen_asr_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_asr_kernels_neon.o: qwen_asr_kernels_neon.c qwen_asr_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_asr_kernels_avx.o: qwen_asr_kernels_avx.c qwen_asr_kernels_impl.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_asr_tokenizer.o: qwen_asr_tokenizer.c qwen_asr_tokenizer.h qwen_asr_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

qwen_asr_safetensors.o: qwen_asr_safetensors.c qwen_asr_safetensors.h
	$(CC) $(CFLAGS) -c -o $@ $<

deps/ae/%.o: deps/ae/%.c deps/ae/ae.h deps/ae/anet.h deps/ae/monotonic.h
	$(CC) $(CFLAGS) -Ideps/ae -c -o $@ $<

# =============================================================================
clean:
	rm -f $(OBJS) pplx_embed_mlx.o main.o $(TARGET)
