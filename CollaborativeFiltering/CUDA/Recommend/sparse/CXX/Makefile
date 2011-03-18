CXX=ccache g++
#OPT=-g -Wall -O3 -ffast-math -funroll-loops 
#OPT=-g -Wall -O3 -funroll-loops 
OPT=-g -Wall -O3 -ffast-math
#OPT=-g -Wall -O3

SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)

recommend: $(OBJS) Makefile
	$(CXX) $(OPT) -o $@ $(OBJS) -lm -ljson

.cpp.o:
	$(CXX) $(OPT) -c -o $@ $<

clean:
	rm *.o *~ recommend
