ROOT := ../..
COMPONENT := server
include $(ROOT)/build.mk

LIBFFWD := $(ROOT)/libffwd
AE         := $(ROOT)/deps/ae

BIN := $(OUTDIR)/ffwd-server
SRCS := base64.c encode.c handlers.c http.c json.c main.c models.c sbuf.c \
        schedule.c server.c util.c
OBJS := $(call objs,$(SRCS))
DEPS := $(call deps,$(OBJS))

TEST_BINS := test_base64 test_sbuf test_encode test_json test_schedule test_models
TEST_OBJS := $(filter-out $(OBJDIR)/main.o,$(OBJS))

LIBS := $(OUTDIR)/libffwd.a $(OUTDIR)/libae.a
LDLIBS := $(CJSON_LDFLAGS) $(BACKEND_LDFLAGS) -lm

CFLAGS += -std=gnu11 -pthread
CPPFLAGS += -I. -I$(LIBFFWD) -I$(AE) $(CJSON_CFLAGS)

$(OBJDIR)/main.o: CFLAGS += $(VERSION_CFLAGS)

.PHONY: all clean test

all: $(BIN)

$(eval $(call link_program,$(BIN),$(OBJS),$(LIBS),$(LDLIBS)))

test: $(TEST_BINS)
	@rc=0; for t in $(TEST_BINS); do ./$$t || rc=1; done; \
	  if [ $$rc -eq 0 ]; then echo "server ok"; else echo "server not ok"; exit 1; fi

$(TEST_BINS): %: %.c test_util.h $(TEST_OBJS) $(LIBS)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(TEST_OBJS) $(LIBS) $(LDLIBS) $(LDFLAGS_OPT) -o $@

$(OUTDIR)/libffwd.a:
	$(MAKE) -C $(LIBFFWD)

$(OUTDIR)/libae.a:
	$(MAKE) -C $(AE)

clean:
	$(RM) -r $(OBJDIR) $(BIN) $(BIN).dSYM \
	    $(TEST_BINS) $(addsuffix .dSYM,$(TEST_BINS))
	$(MAKE) -C $(AE) clean

-include $(DEPS)
