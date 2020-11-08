#include "sim_mpi.h"

int world_size;
int my_rank;
int rows_per_slaves;

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
    cerr << "RANK: " << my_rank;
    for (int i = 0; i < lattice.loops; ++i) {
        lattice.avgEnergy[i] = lattice.calculate_energy();
        lattice.avgMag[i] = lattice.calculate_magnetism();
        spins(lattice);
    }

    return EXIT_SUCCESS;
}

/**
 * Initialisation of clocks to measure the runtime of the different
 * parallelization techniques.
 * @param lattice The Surface object containing the lattice configuration and functions.
 * @return The EXIT_STATUS of the simulation.
 */
int initialise(Surface lattice) {
    int status;

    auto StartTime = std::chrono::high_resolution_clock::now();
    if (my_rank == 0) {
        lattice.clear();
        lattice.save();
    } else {
        status = simulate(lattice);
    }

    auto FinishTime = std::chrono::high_resolution_clock::now();

    if (my_rank == 0) {
        lattice.complete = true;
        lattice.save();
        auto TotalTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishTime - StartTime);
        //cout << lattice.name << ":" << endl;
        //cout << "Total time: " << std::setw(12) << TotalTime.count() << " us" << endl;
    }




    return status;
}

/**
 * Main function of the application. Handles boundry and error checking.
 * Will initiate the simulation if everything is correct.
 * @param argc The number of arguments.
 * @param argv The string representation array of the arguments.
 * @return The EXIT_STATUS of the simulation.
 */
int main(int argc, char* argv[]) {
    int status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    rows_per_slaves = world_size - 1;


    if (argc < 4 || argc > 5) {
        std::cout << "Usage: ./ising n size temperature {output}" << std::endl;
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        std::cout << "'n' has to be a positive integer" << std::endl;
        return EXIT_FAILURE;
    }

    int rows = atoi(argv[2]);
    if (rows <= 0) {
        std::cout << "'rows' has to be a positive integer" << std::endl;
        return EXIT_FAILURE;
    }

    Surface lattice(argv[0], n, rows, strtod(argv[3], nullptr));

    if (argc == 5) {
        lattice.out = true;
        lattice.outName = argv[4];
    }

    if (my_rank == 0) {
        lattice.output_surface(cout);
        for (int i = 1; i < world_size; ++i) {
            for (int y = 0; y < lattice.size; ++y) {
                MPI_Send(lattice.surface[y], lattice.size, MPI_INT, i, y, MPI_COMM_WORLD);
            }
        }


    } else {
        for (int y = 0; y < lattice.size; ++y) {
            MPI_Recv(lattice.surface[y], lattice.size, MPI_INT, 0, y, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        lattice.output_surface(cerr);
    }

    status = initialise(lattice);

    MPI_Finalize();
    return status;
}
