CXX = g++
CXXFLAGS = -O3 -pthread

all: pi.out

pi.out: pi.cpp
	$(CXX) $(CXXFLAGS) pi.cpp -o pi.out

clean:
	rm -f pi.out
