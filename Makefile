include build.mk

# Top-level build:
# Root owns BACKEND/MODE and exports the resolved flags through build.mk.

LIBS  := libffwd
TOOLS := tools/cli tools/server

.PHONY: all libs tools test clean debug profile release blas cpu mlx cuda gpu \
        bench bench-model bench-tokenizer bench-server-utils $(LIBS) $(TOOLS)

all: libs tools
	@ln -sf $(BUILDREL)/ffwd-cli ./ffwd-cli
	@ln -sf $(BUILDREL)/ffwd-server ./ffwd-server

libs:  $(LIBS)
tools: $(TOOLS)

$(LIBS):
	$(MAKE) -C $@

tools/cli: | libffwd
	$(MAKE) -C $@

tools/server: | libffwd
	$(MAKE) -C $@

# The current unit suite is CPU-internal; GPU checks are explicit driver targets.
test:
	$(MAKE) -C tests BACKEND=blas test
	$(MAKE) -C tools/server BACKEND=blas test

clean:
	$(RM) -r build
	@for d in $(TOOLS) tests bench; do $(MAKE) -C $$d clean; done
	$(RM) -f ./ffwd-cli ./ffwd-server ffwd.exports ffwd.map

blas:
	$(MAKE) BACKEND=blas MODE=$(MODE) all

cpu: blas

mlx:
	$(MAKE) BACKEND=mlx MODE=$(MODE) all

cuda:
	$(MAKE) BACKEND=cuda MODE=$(MODE) all

ifeq ($(OS),Darwin)
gpu: mlx
else
gpu: cuda
endif

release:
	$(MAKE) MODE=release BACKEND=$(BACKEND) all

debug:
	$(MAKE) MODE=debug BACKEND=$(BACKEND) all

profile:
	$(MAKE) MODE=profile BACKEND=$(BACKEND) all

bench bench-model bench-tokenizer bench-server-utils:
	$(MAKE) -C bench $@
