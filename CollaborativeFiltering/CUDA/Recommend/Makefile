# NFLAG = -D__DEVICE_EMULATION__  -deviceemu

CC = gcc
NVCC = nvcc
RM = /bin/rm
PROG = recommend

OBJS = recommend.o
CFLAGS = -O3 --compiler-options -fno-inline
#CFLAGS+=-deviceemu -D__DEVICE_EMULATION__ 
LDFLAGS = ${CFLAGS}

M_ARCH = $(shell uname -m)

# for Device Code
CUDA_PATH = /usr/local/cuda
CUDA_SDK_PATH = ${HOME}/NVIDIA_GPU_Computing_SDK/C
ifeq ($(M_ARCH), x86_64)
LDFLAGS += -L${CUDA_PATH}/lib64
else
LDFLAGS += -L${CUDA_PATH}/lib
endif
LDFLAGS += -L${CUDA_SDK_PATH}/lib
LDFLAGS += -lcudart -lcutil
NFLAG += ${CFLAGS} -I${CUDA_SDK_PATH}/common/inc
NFLAG += -maxrregcount 32

all : ${PROG}

${PROG} : ${OBJS}
	${CC} -o $@ ${OBJS} ${LDFLAGS}

%.o : %.c
	${CC} -c ${CFLAGS} $<

%.o : %.cu
	${NVCC} -c ${NFLAG} $<

clean :
	${RM} -f ${PROG} ${OBJS} *.o core

