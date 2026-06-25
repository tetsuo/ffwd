ROOT := ../..
COMPONENT := cli
include $(ROOT)/build.mk

.DEFAULT_GOAL := all

LIBFFWD := $(ROOT)/libffwd
YYJSON  := $(ROOT)/deps/yyjson

BIN := $(OUTDIR)/ffwd-cli
SRCS := main.c
LIBS := $(OUTDIR)/libffwd.a $(OUTDIR)/libyyjson.a
LDLIBS := $(BACKEND_LDFLAGS) -lm

CFLAGS += -std=gnu11 -pthread
CPPFLAGS += -I. -I$(LIBFFWD)

$(OBJDIR)/main.o: CFLAGS += $(VERSION_CFLAGS)

$(OUTDIR)/libffwd.a:
	$(MAKE) -C $(LIBFFWD)

$(OUTDIR)/libyyjson.a:
	$(MAKE) -C $(YYJSON)

$(eval $(program))
