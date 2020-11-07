#include "cuda.h"

Surface* mLattice;

// Error handling function from "cosc3500/cuda/example1-gpu.cu"
void checkError(cudaError_t e) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
        abort();
    }
}

__global__
void spins(Surface* lattice) {
    for (int j = 0; j < lattice.size; ++j) {
        for (int k = 0; k < lattice.size; ++k) {
            int coords[2] = {j,k};
            lattice.calculate_spin(coords);
        }
    }
}


/**
 */
int simulate(Surface lattice) {
    for (int i = 0; i < lattice.loops; ++i) {
        lattice.avgEnergy[i] = lattice.calculate_energy();
        lattice.avgMag[i] = lattice.calculate_magnetism();


        checkError(cudaMalloc((void **)&mLattice, sizeof(Surface)));
        checkError(cudaMemcpy(mLattice, lattice, sizeof(Surface), cudaMemcpyHostToDevice));

        cudaFree(mLattice);

        spins(&lattice);
    }
    return EXIT_SUCCESS;
}