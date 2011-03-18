# NFLAG = -D__DEVICE_EMULATION__  -deviceemu

CC = ccache gcc-4.3
CXX = ccache g++-4.3
NVCC = nvcc --compiler-bindir .
RM = /bin/rm
PROG = recommend

CXX_SRCS = $(wildcard *.cpp)
CU_SRCS = $(wildcard *.cu)
OBJS = $(CXX_SRCS:.cpp=.o)  $(CU_SRCS:.cu=.o) 

CFLAGS = -O3 -ffast-math
#CFLAGS+=-deviceemu -D__DEVICE_EMULATION__ 
LDFLAGS = ${CFLAGS} -lm -ljson

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
NFLAG += -O3 -use_fast_math -I${CUDA_SDK_PATH}/common/inc
#NFLAG += -O3 --compiler-options -fno-inline -use_fast_math -I${CUDA_SDK_PATH}/common/inc
#NFLAG +=-deviceemu -D__DEVICE_EMULATION__ 
#NFLAG += -maxrregcount 32

all : ${PROG}

${PROG} : ${OBJS}
	${CC} -o $@ ${OBJS} ${LDFLAGS}

%.o : %.c
	${CC} -c ${CFLAGS} $<

%.o : %.cpp
	${CXX} -c ${CFLAGS} $<

%.o : %.cu
	${NVCC} -c ${NFLAG} $<

clean :
	${RM} -f ${PROG} ${OBJS} *.o core

