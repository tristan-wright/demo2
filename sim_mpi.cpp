#include <mpi.h>
#include "sim_mpi.h"

int spins(Surface lattice) {
    for (int j = 0; j < lattice.size; ++j) {
        for (int k = 0; k < lattice.size; ++k) {
            int coords[2] = {j,k};
            lattice.calculate_spin(coords);
        }
    }

    return EXIT_SUCCESS;
}

/**
 */
int simulate(Surface lattice) {
    MPI_Init(nullptr, nullptr);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        for (int i = 0; i < lattice.loops; ++i) {
            lattice.avgEnergy[i] = lattice.calculate_energy();
            lattice.avgMag[i] = lattice.calculate_magnetism();
            spins(lattice);

        }
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}