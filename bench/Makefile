ROOT := ..
include ../build.mk

CFLAGS += -std=c11

BLAS_LIBS := $(HOST_BLAS_LDFLAGS)
JSON_LIBS := $(YYJSON_LDFLAGS)

# Benchmarks link the blas libffwd archive from the per-config build tree and
# include its headers.
LIBFFWD_DIR := ../libffwd
LIBFFWD_A   := $(ROOT)/build/$(MODE)/blas/libffwd.a
SERVER_DIR     := ../tools/server

RESULTS := results
# Commit, a -dirty marker when the tree has uncommitted changes, and a timestamp,
# so repeated runs at the same commit do not overwrite each other.
STAMP ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)$(shell git diff --quiet 2>/dev/null || echo -dirty)-$(shell date +%Y%m%d-%H%M%S)

BENCHES := bench_kernels bench_tokenizer bench_server_utils bench_model \
           bench_late_rerank

.PHONY: all bench bench-model bench-tokenizer bench-server-utils clean \
        libffwd-blas

all: $(BENCHES)

# The archive is built with a backend-specific flag, so a phony target (not a
# file rule) drives the sub-make; the benchmarks list it as an order prerequisite
# and link the resulting $(LIBFFWD_A).
libffwd-blas:
	$(MAKE) -C $(LIBFFWD_DIR) BACKEND=blas

$(eval $(call c_program_deps,bench_kernels,bench_kernels.c,bench.h libffwd-blas,$(LIBFFWD_A),-pthread -I$(LIBFFWD_DIR),$(BLAS_LIBS) $(JSON_LIBS) -lm))

$(eval $(call c_program_deps,bench_tokenizer,bench_tokenizer.c,libffwd-blas,$(LIBFFWD_A),-I$(LIBFFWD_DIR),$(JSON_LIBS) -lm))

# The server utilities (base64, sbuf, util) depend on no library, so the bench
# compiles those sources directly here and keeps the build self-contained.
SERVER_UTIL_SRCS := $(SERVER_DIR)/base64.c $(SERVER_DIR)/sbuf.c $(SERVER_DIR)/util.c
$(eval $(call c_program_deps,bench_server_utils,bench_server_utils.c $(SERVER_UTIL_SRCS),bench.h,,-I$(SERVER_DIR),-lm))

$(eval $(call c_program_deps,bench_model,bench_model.c,bench.h libffwd-blas,$(LIBFFWD_A),-pthread -I$(LIBFFWD_DIR),$(BLAS_LIBS) $(JSON_LIBS) -lm))

$(eval $(call c_program_deps,bench_late_rerank,bench_late_rerank.c,libffwd-blas,$(LIBFFWD_A),-pthread -I$(LIBFFWD_DIR),$(BLAS_LIBS) $(JSON_LIBS) -lm))

# MLX rerank benchmark: the blas engine archive for the CPU reference paths plus
# mlx.c for the device API. The archive calls cblas, so the bench links
# Accelerate/OpenBLAS itself ($(HOST_BLAS_LDFLAGS)) on top of Metal/MLX. macOS only.
MLX_DRIVER_CFLAGS := -pthread -DCHECK_MLX -I$(MLXC_PREFIX)/include -I$(LIBFFWD_DIR) $(HOST_BLAS_CFLAGS)
MLX_DRIVER_LDLIBS := $(MLX_BACKEND_LDFLAGS) $(HOST_BLAS_LDFLAGS) $(JSON_LIBS) -lm
$(eval $(call c_program_deps,bench_late_rerank_mlx,bench_late_rerank.c $(LIBFFWD_DIR)/mlx.c,libffwd-blas,$(LIBFFWD_A),$(MLX_DRIVER_CFLAGS),$(MLX_DRIVER_LDLIBS)))

bench: bench_kernels
	@mkdir -p $(RESULTS)
	./bench_kernels --json $(RESULTS)/kernels-$(STAMP).json
	@echo "compare: ./benchstat.py $(RESULTS)/OLD.json $(RESULTS)/NEW.json"

bench-server-utils: bench_server_utils
	@mkdir -p $(RESULTS)
	./bench_server_utils --json $(RESULTS)/server-utils-$(STAMP).json
	@echo "compare: ./benchstat.py $(RESULTS)/OLD.json $(RESULTS)/NEW.json"

bench-model: bench_model
	@test -n "$(MODEL_DIR)" || { echo "usage: make bench-model MODEL_DIR=/path/to/model-dir"; exit 1; }
	@mkdir -p $(RESULTS)
	./bench_model "$(MODEL_DIR)" --json $(RESULTS)/model-$(STAMP).json
	@echo "compare: ./benchstat.py $(RESULTS)/OLD.json $(RESULTS)/NEW.json"

bench-tokenizer: bench_tokenizer
	@echo "run: ./bench_tokenizer <model_dir-or-vocab.json> <runs> TEXT..."

clean:
	$(RM) -r $(BENCHES) bench_late_rerank_mlx *.dSYM
