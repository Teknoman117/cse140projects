CC = gcc -std=gnu99
CXX = g++ -std=c++11

CFLAGS = -g -msse4  -pipe -fno-omit-frame-pointer -fopenmp
#CFLAGS = -pipe -fno-omit-frame-pointer
CXXFLAGS = $(CFLAGS)
LDFLAGS = -fopenmp

all:	bench-naive bench-optimize test

#naive
bench-naive: benchmark.o sgemm-naive.o
	$(CXX) $(LDFLAGS) -o $@ benchmark.o sgemm-naive.o

#optimize
bench-optimize: benchmark.o sgemm-optimize.o
	$(CXX) $(LDFLAGS) -o $@ benchmark.o sgemm-optimize.o

test: test.o
	$(CXX) $(LDFLAGS) -o $@ test.o

%.o: %.c
	$(CC) -c $(CFLAGS) $<

%.o: %.cxx
	$(CXX) -c $(CXXFLAGS) $<

clean:
	rm -f *~ bench-naive bench-optimize test *.o
