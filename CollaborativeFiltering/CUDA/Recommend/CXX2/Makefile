CXX=ccache g++
OPT=-g -Wall -O3 -ffast-math -funroll-loops 
#OPT=-g -Wall -O3 -funroll-loops 
#OPT=-g -Wall -O3 -ffast-math
#OPT=-g -Wall -O3

recommend: recommend.cpp Makefile
	$(CXX) $(OPT) -o $@ $< -lm -ljson
