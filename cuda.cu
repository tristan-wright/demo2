#include "cuda.h"

// Error handling function from "cosc3500/cuda/example1-gpu.cu"
void checkError(cudaError_t e)
{
    if (e != cudaSuccess)
    {
        std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
        abort();
    }
}

/**
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