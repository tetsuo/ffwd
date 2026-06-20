ROOT := ..
include ../build.mk

CFLAGS += -std=c11

BLAS_LIBS := $(HOST_BLAS_LDFLAGS)

# Every test links the blas libffwd archive (built into the per-config build
# tree by the libffwd-blas phony) and includes its headers; only the fixture
# headers are local to this directory.
LIBFFWD_DIR := ../libffwd
LIBFFWD_A   := $(ROOT)/build/$(MODE)/blas/libffwd.a

# Tokenizer and safetensors tests pull self-contained objects (no thread/BLAS
# link flags); kernel and model tests pull the math kernels and forward pass.
TOK_TESTS   := test_bpe test_wordpiece test_sentencepiece
ST_TESTS    := test_safetensors
KERNEL_TESTS := test_kernels
FFWD_TESTS := test_cls test_late test_workspace test_bf16_model test_qwen2 \
               test_qwen3 test_xlm_roberta
# test_kernels_generic recompiles the kernels with no USE_BLAS to exercise the
# scalar/SIMD paths (no cblas); test_kernels covers the blas-built kernels.
KERNEL_SRCS := $(addprefix $(LIBFFWD_DIR)/,act.c gemm.c norm.c rope.c attn.c threadpool.c generic.c avx.c neon.c)
TESTS := $(TOK_TESTS) $(ST_TESTS) $(KERNEL_TESTS) test_kernels_generic $(FFWD_TESTS)

GEN_FIXTURE_BIN := gen_fixture

PARITY_TOK_DRIVERS := tokenizer_dump wordpiece_dump
PARITY_FFWD_DRIVERS := bert_dump contextual_batch
INTEGRATION_DRIVERS := late_check
FFWD_DRIVERS := $(PARITY_FFWD_DRIVERS) $(INTEGRATION_DRIVERS)
PARITY_DRIVERS := $(PARITY_TOK_DRIVERS) $(FFWD_DRIVERS)

.PHONY: all test coverage clean libffwd-blas \
        parity-tokenizer-driver parity-wordpiece-driver parity-bert-driver \
        parity-context-driver parity-context-driver-mlx late-check-driver \
        late-check-driver-mlx

all: $(TESTS) $(GEN_FIXTURE_BIN)

test: $(TESTS)
	@rc=0; for t in $(TESTS); do ./$$t || rc=1; done; \
	  if [ $$rc -eq 0 ]; then echo "tests ok"; else echo "tests not ok"; fi; \
	  exit $$rc

# The archive is backend-specific, so a phony (not a file rule) drives the
# sub-make; tests list it as an order prerequisite and link $(LIBFFWD_A).
libffwd-blas:
	$(MAKE) -C $(LIBFFWD_DIR) BACKEND=blas

parity-tokenizer-driver: tokenizer_dump
parity-wordpiece-driver: wordpiece_dump
parity-bert-driver: bert_dump
parity-context-driver: contextual_batch
parity-context-driver-mlx: contextual_batch_mlx
late-check-driver: late_check
late-check-driver-mlx: late_check_mlx

$(foreach t,$(TOK_TESTS),\
    $(eval $(call c_program_deps,$(t),$(t).c,libffwd-blas,$(LIBFFWD_A),-I$(LIBFFWD_DIR),-lm)))

$(foreach t,$(ST_TESTS),\
    $(eval $(call c_program_deps,$(t),$(t).c,libffwd-blas,$(LIBFFWD_A),-I$(LIBFFWD_DIR),)))

$(foreach t,$(KERNEL_TESTS),\
    $(eval $(call c_program_deps,$(t),$(t).c,libffwd-blas,$(LIBFFWD_A),-pthread -I$(LIBFFWD_DIR),$(BLAS_LIBS) -lm)))

# Generic (no USE_BLAS) kernel build: compile the kernel sources directly so the
# scalar/SIMD paths are linked instead of cblas; no Accelerate/OpenBLAS needed.
$(eval $(call c_program_deps,test_kernels_generic,test_kernels.c $(KERNEL_SRCS),,,-pthread -I$(LIBFFWD_DIR),-lm))

$(foreach t,$(FFWD_TESTS),\
    $(eval $(call c_program_deps,$(t),$(t).c,libffwd-blas,$(LIBFFWD_A),-pthread -I$(LIBFFWD_DIR),$(BLAS_LIBS) -lm)))

$(foreach t,$(PARITY_TOK_DRIVERS),\
    $(eval $(call c_program_deps,$(t),$(t).c,libffwd-blas,$(LIBFFWD_A),-I$(LIBFFWD_DIR),-lm)))

$(foreach t,$(FFWD_DRIVERS),\
    $(eval $(call c_program_deps,$(t),$(t).c,libffwd-blas,$(LIBFFWD_A),-pthread -I$(LIBFFWD_DIR),$(BLAS_LIBS) -lm)))

# The MLX driver variants reuse the BLAS engine archive for the CPU reference
# paths (e.g. ffwd_late_maxsim) and add mlx.c for the device API. The archive
# calls cblas, so the driver links Accelerate/OpenBLAS itself ($(HOST_BLAS_LDFLAGS))
# on top of the Metal/MLX libraries. macOS only, built on demand.
MLX_DRIVER_CFLAGS := -pthread -DCHECK_MLX -I$(MLXC_PREFIX)/include -I$(LIBFFWD_DIR) $(HOST_BLAS_CFLAGS)
MLX_DRIVER_LDLIBS := $(MLX_BACKEND_LDFLAGS) $(HOST_BLAS_LDFLAGS) -lm

$(eval $(call c_program_deps,contextual_batch_mlx,contextual_batch.c $(LIBFFWD_DIR)/mlx.c,libffwd-blas,$(LIBFFWD_A),$(MLX_DRIVER_CFLAGS),$(MLX_DRIVER_LDLIBS)))

$(eval $(call c_program_deps,late_check_mlx,late_check.c $(LIBFFWD_DIR)/mlx.c,libffwd-blas,$(LIBFFWD_A),$(MLX_DRIVER_CFLAGS),$(MLX_DRIVER_LDLIBS)))

$(eval $(call c_program_deps,$(GEN_FIXTURE_BIN),gen_fixture.c,model_fixture.h tiny_model.h tok_fixture.h,,-I.,-lm))

# Line coverage (needs clang plus llvm-cov / llvm-profdata; xcrun on macOS).
# The compile, run, and report steps all run from the repo root so the recorded
# source paths stay clean (libffwd/x.c, tests/x.c) with no leading '..'; that
# is what keeps the generated HTML's css/js links resolving correctly. Each test
# is compiled at -O0 with the engine sources built in and instrumented, run to
# emit a .profraw, then merged into a text summary and an HTML report under
# tests/coverage/html. Only library sources are reported.
COV_ROOT   := ..
COV_DIR    := coverage
COV_OUT    := tests/$(COV_DIR)
COV_FLAGS  := -fprofile-instr-generate -fcoverage-mapping -O0
COV_CFLAGS := -Wall -Wextra -Ilibffwd -Itests -pthread $(HOST_BLAS_CFLAGS) $(COV_FLAGS)
COV_LDLIBS := $(BLAS_LIBS) -lm
# Report library sources only: drop the test harnesses (test_*), the fixture
# headers, and system headers.
COV_IGNORE := -ignore-filename-regex='test_|_fixture|tiny_model|/opt/|/usr/'
# The blas engine, root-relative: every libffwd source except the GPU backends.
ENGINE_COV_SRCS := $(filter-out libffwd/mlx.c libffwd/backend_mlx.c libffwd/backend_cuda.c,$(patsubst $(LIBFFWD_DIR)/%,libffwd/%,$(wildcard $(LIBFFWD_DIR)/*.c)))
COV_BINS   := $(addprefix $(COV_OUT)/,$(filter-out test_kernels_generic,$(TESTS)))

ifeq ($(OS),Darwin)
LLVM_PROFDATA := xcrun llvm-profdata
LLVM_COV      := xcrun llvm-cov
else
LLVM_PROFDATA := llvm-profdata
LLVM_COV      := llvm-cov
endif

# Compile (from the repo root) one instrumented test with the engine built in,
# then run it. cov_min links a minimal source subset $(2) for a test that
# supplies its own globals (the full engine would double-define them).
define cov_one
cd $(COV_ROOT) && $(CC) $(COV_CFLAGS) -o $(COV_OUT)/$(1) tests/$(1).c $(ENGINE_COV_SRCS) $(COV_LDLIBS) && \
LLVM_PROFILE_FILE=$(COV_OUT)/$(1).profraw $(COV_OUT)/$(1)
endef

define cov_min
cd $(COV_ROOT) && $(CC) $(COV_CFLAGS) -o $(COV_OUT)/$(1) tests/$(1).c $(2) $(COV_LDLIBS) && \
LLVM_PROFILE_FILE=$(COV_OUT)/$(1).profraw $(COV_OUT)/$(1)
endef

coverage:
	$(RM) -r $(COV_DIR)
	@mkdir -p $(COV_DIR)
	$(call cov_min,test_bpe,libffwd/bpe.c)
	$(call cov_one,test_wordpiece)
	$(call cov_one,test_sentencepiece)
	$(call cov_one,test_safetensors)
	$(call cov_one,test_kernels)
	$(call cov_one,test_cls)
	$(call cov_one,test_late)
	$(call cov_one,test_workspace)
	$(call cov_one,test_bf16_model)
	$(call cov_one,test_qwen2)
	$(call cov_one,test_qwen3)
	$(call cov_one,test_xlm_roberta)
	cd $(COV_ROOT) && $(LLVM_PROFDATA) merge -sparse $(COV_OUT)/*.profraw -o $(COV_OUT)/tests.profdata
	cd $(COV_ROOT) && $(LLVM_COV) report -instr-profile=$(COV_OUT)/tests.profdata $(COV_IGNORE) \
	    $(addprefix -object ,$(COV_BINS))
	cd $(COV_ROOT) && $(LLVM_COV) show -format=html -output-dir=$(COV_OUT)/html \
	    -instr-profile=$(COV_OUT)/tests.profdata -show-line-counts-or-regions \
	    $(COV_IGNORE) $(addprefix -object ,$(COV_BINS))
	@echo "HTML report: $(COV_OUT)/html/index.html"

clean:
	$(RM) -r $(TESTS) $(PARITY_DRIVERS) contextual_batch_mlx late_check_mlx \
	    *.dSYM $(GEN_FIXTURE_BIN) $(COV_DIR)
