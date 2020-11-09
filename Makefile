SOURCES = surface.cpp
FLAGS = -O

all:
	g++ -o ising_serial $(SOURCES) sim_serial.cpp $(FLAGS)
	g++ -fopenmp -o ising_omp $(SOURCES) sim_omp.cpp $(FLAGS)
	mpic++ -g2 -o ising_mpi $(SOURCES) sim_mpi.cpp

clean:
	rm ising_serial ising_omp ising_mpi