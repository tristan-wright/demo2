SOURCES = main.cpp surface.cpp serial.cpp

all:
	g++ -o ising $(SOURCES) -O

clean:
	rm ising