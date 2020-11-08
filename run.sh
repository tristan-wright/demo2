export OMP_NUM_THREADS=4

make clean
make
#cat dev/null > test.out
./ising_serial 2 200 2 test.out
./ising_omp 2 200 2 test2.out
./ising_mpi 2 200 2 test2.out
#python3 display.py