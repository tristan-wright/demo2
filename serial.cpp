#include "serial.h"

/**
 * Runs the simulation keeping track of various states.
 * @param lattice The lattice to simulate.
 * @return The exit state of the application.
 */
int simulate(Surface lattice) {
    for (int i = 0; i < lattice.loops; ++i) {
        lattice.avgEnergy[i] = lattice.calculate_energy();
        lattice.avgMag[i] = lattice.calculate_magnetism();

        for (int j = 0; j < lattice.size; ++j) {
            for (int k = 0; k < lattice.size; ++k) {
                int coords[2] = {j,k};
                lattice.calculate_spin(coords);
            }
        }
    }
    return EXIT_SUCCESS;
}