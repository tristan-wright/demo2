#include <iostream>
#include <cstdlib>
#include <random>
#include <fstream>

#include "serial.h"
#include "surface.h"

/**
 * Main function of the application. Handles boundry and error checking.
 * Will initiate the simulation if everything is correct.
 * @param argc The number of arguments.
 * @param argv The string representation array of the arguments.
 * @return The exit status of the simulation.
 */
int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        std::cout << "Usage: ./ising n size temperature {output}" << std::endl;
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        std::cout << "'n' has to be a positive integer" << std::endl;
        return EXIT_FAILURE;
    }

    Surface lattice(n, atoi(argv[2]), strtod(argv[3], nullptr));

    if (argc == 5) {
        lattice.out = true;
        lattice.outName = argv[4];
    }

    return simulate(lattice);
}
