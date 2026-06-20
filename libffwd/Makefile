ROOT := ..
COMPONENT := libffwd
include ../build.mk

LIB := ffwd
STATIC := $(OUTDIR)/lib$(LIB).a
SHARED := $(OUTDIR)/lib$(LIB).$(SHARED_EXT)

CFLAGS += -std=c11 -pthread $(BACKEND_CFLAGS)
CPPFLAGS += -I.
SHARED_FLAGS += -pthread
# Restrict the shared library's exported symbols to the public ABI.
SHARED_LDFLAGS += $(EXPORT_FLAGS)

# Tokenizer, config, safetensors reader, and L2/cosine helpers: every backend.
COMMON_SRCS := config.c vec.c tokenize.c bpe.c sentencepiece.c wordpiece.c safetensors.c

# CPU math kernels and SIMD dispatch. generic.c is always built; avx.c and
# neon.c each guard their entire body by arch (__AVX2__ / __ARM_NEON), so the
# off-target file compiles to an empty translation unit.
KERNEL_SRCS := act.c gemm.c norm.c rope.c attn.c threadpool.c generic.c avx.c neon.c

BLAS_SRCS := backend_blas.c model.c model_blas.c forward.c late.c late_blas.c alloc.c \
             $(COMMON_SRCS) $(KERNEL_SRCS)
MLX_SRCS  := backend_mlx.c mlx.c $(COMMON_SRCS)
CUDA_SRCS := backend_cuda.c model.c late.c alloc.c $(COMMON_SRCS) $(KERNEL_SRCS)

ifeq ($(BACKEND),mlx)
    SRCS := $(MLX_SRCS)
    CU_SRCS :=
else ifeq ($(BACKEND),cuda)
    SRCS := $(CUDA_SRCS)
    CU_SRCS := cuda.cu
else
    SRCS := $(BLAS_SRCS)
    CU_SRCS :=
endif

OBJS := $(call objs,$(SRCS) $(CU_SRCS))
DEPS := $(call deps,$(OBJS))

.PHONY: all clean

all: $(STATIC) $(SHARED)

$(eval $(call static_archive,$(STATIC),$(OBJS)))
$(eval $(call shared_library,$(SHARED),$(OBJS),,$(BACKEND_LDFLAGS) -lm))
# The export map is generated from the public ABI (the FFWD_API declarations in
# ffwd.h) instead of being committed, so it cannot drift from the header. It is
# regenerated when the header or the generator changes, and the shared library
# relinks when it does.
$(EXPORT_FILE): ffwd.h $(ROOT)/devtools/gen_exports.py
	python3 $(ROOT)/devtools/gen_exports.py --outdir $(ROOT)
$(SHARED): $(EXPORT_FILE)

# Removes only this MODE/BACKEND's objects and artifacts; the root `make clean`
# removes the whole build/ tree.
clean:
	$(RM) -r $(OBJDIR) $(STATIC) $(SHARED) $(SHARED).dSYM

-include $(DEPS)
