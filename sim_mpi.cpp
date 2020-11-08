#include <mpi.h>
#include "sim_mpi.h"

/**
 */
int simulate(Surface lattice) {
    MPI_Init(nullptr, nullptr);
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
    MPI_Finalize();
    return EXIT_SUCCESS;
}