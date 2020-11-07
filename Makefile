SOURCES = main.cpp surface.cpp
FLAGS = -O

all:
	g++ -o ising_serial $(SOURCES) serial.cpp $(FLAGS)
	g++ -fopenmp -o ising_omp $(SOURCES) omp.cpp $(FLAGS)
	#nvcc -o ising_cuda $(SOURCES) cuda.cu

clean:
	rm ising_serial ising_omp ising_cuda