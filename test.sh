make clean
make
cat dev/null > test.out
time ./ising 10 200 2 test.out
python3 display.py