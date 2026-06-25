ROOT := ../..
COMPONENT := yyjson
include $(ROOT)/build.mk

LIB := yyjson

SRCS := yyjson.c
HEADERS := yyjson.h

# yyjson has its own IEEE-754 number parser and printer, so it must build with
# standard floating-point semantics. Strip the project-wide reassociation flags
# (used by the math kernels) from this unit; they would change number output.
FP_UNSAFE := -fno-math-errno -ffp-contract=fast -fno-trapping-math \
             -fno-signed-zeros -fassociative-math -freciprocal-math
CFLAGS := $(filter-out $(FP_UNSAFE),$(CFLAGS)) -std=c11

$(eval $(simple_library))
