ROOT := ../..
COMPONENT := ae
include $(ROOT)/build.mk

LIB := ae

# ae.c includes the platform poller; ae_kqueue.c/ae_epoll.c/ae_select.c are not
# separate translation units.
SRCS := ae.c anet.c monotonic.c
HEADERS := ae.h anet.h monotonic.h

CFLAGS += -std=gnu11 -DREDIS_STATIC=

$(eval $(simple_library))
