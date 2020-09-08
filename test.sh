cat dev/null > test.out
time ./cmake-build-debug/ising 10 200 2.2 test.out
python3 display.py