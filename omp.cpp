#include "omp.h"

/**
 */
int simulate(Surface lattice) {
    for (int i = 0; i < lattice.loops; ++i) {
        lattice.avgEnergy[i] = lattice.calculate_energy();
        lattice.avgMag[i] = lattice.calculate_magnetism();

        #pragma omp parallel for
        for (int j = 0; j < lattice.size; ++j) {
            for (int k = 0; k < lattice.size; ++k) {
                int coords[2] = {j,k};
                lattice.calculate_spin(coords);
            }
        }
    }
    return EXIT_SUCCESS;
}