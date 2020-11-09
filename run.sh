export OMP_NUM_THREADS=4

make
./ising_serial 2 200 2 test.out
./ising_omp 2 200 2 test2.out
mpirun -n 5 ./ising_mpi 2 200 2 test3.out
python3 display.py test3.out