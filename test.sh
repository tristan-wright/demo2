make clean
make
cat dev/null > test.out
time ./ising 50 200 2.2 test.out
python3 display.py