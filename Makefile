SOURCES = main.cpp surface.cpp

all:
	g++ -o ising_serial $(SOURCES) serial.cpp -O
	g++ -fopenmp -o ising_omp $(SOURCES) omp.cpp -O

clean:
	rm ising_serial ising_omp