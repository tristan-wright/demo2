#include <iostream>
#include <cstdlib>
#include <random>
#include <fstream>

using namespace std;

class Surface {
public:
    /**
     * Constructor method takes the size and temperature
     * of the surface. It then creates the N^2 surface.
     * @param size The size of the N^2 lattice.
     * @param temp  The temperature of the lattice.
     */
    Surface(int loops, int size, double temp) {
        Surface::size = size;
        Surface::temp = temp;
        Surface::loops = loops;
        surface = create_surface();
        beta = boltzmann * temp;
        avgEnergy = new int[loops];
        avgMag = new int[loops];
    }

    void output_config(ostream &output) const;
    void save();
    int calculate_magnetism();
    int calculate_energy();
    int random_point() const;
    int energy_neighbours(int x, int y);
    int get_state(int x, int y);
    void set_state(int x, int y, int new_state);
    void output_eng_mag(ostream& output) const;
    bool complete = false;

    bool out = false;
    char* outName = nullptr;
    double beta;
    int size;
    int* avgEnergy;
    int* avgMag;
    int loops;

private:
    const double boltzmann = 1.3806503;
    int** create_surface() const;
    void output_surface(ostream &output) const;

    int** surface;
    double temp;
};

/**
* Outputs the spins for each atom that is stored in
* the surface.
* @param surface A pointer to the surface variable.
* @param size The size of the surface.
*/
void Surface::output_surface(ostream& output) const{
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            output << surface[j][i];
            if (j != size-1) {
                output << ' ';
            }
        }
        if (i != size-1) {
            output << "; ";
        }
    }
}

/**
* Uses Bernoulli distribution to determine the get_state
* of the spin for each atom. Either +/- 1.
* @param size The size of the matrix.
* @return
*/
int** Surface::create_surface() const{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dis;
    int** new_surface;
    new_surface = new int*[size];

    for (int i = 0; i < size; i++) {
        new_surface[i] = new int[size];
        for (int j = 0; j < size; j++) {
            if (dis(gen)) {
                new_surface[i][j] = -1;
            } else {
                new_surface[i][j] = 1;
            }
        }
    }
    return new_surface;
}

/**
 * Outputs the surface to a given file.
 * @param filename Name of the output file.
 */
void Surface::save() {
    if (!out) {
        return;
    }
    std::ofstream file;
    file.open(outName, ios_base::app);
    output_surface(file);
    file << "\n";
    if (complete) {
        output_eng_mag(file);
    }
    file.close();
}

/**
 * Outputs the configuration of the environment
 * to the given std::ostream.
 * @param output A std:ostream.
 */
void Surface::output_config(ostream& output) const {
    output << size << ',' << temp << std::endl;
}
/**
 * Outputs the total energy and magnetism of the surface
 * at a given instance.
 * @param output A std:ostream.
 */
void Surface::output_eng_mag(ostream& output) const {
    for (int i = 0; i < loops; ++i) {
        output << avgEnergy[i];
        if (i != loops - 1) {
            output << ' ';
        }
    }
    output << endl;
    for (int j = 0; j < loops; ++j) {
        output << avgMag[j];
        if (j != loops - 1) {
            output << ' ';
        }
    }
    output << endl;
}

/**
 * Finds the current get_state of the vertex in
 * that position.
 * @param x The horizontal coordinate of the vertex.
 * @param y The vertical coordinate of the vertex.
 * @return The current get_state of the vertex.
 */
int Surface::get_state(int x, int y) {
    if (x < 0) {
        x = size + x;
    } else if (y < 0) {
        y = size + y;
    }
    return surface[y][x];
}

/**
 * Sets the spin state of a given coordinate.
 * @param x The x position of the coordinate.
 * @param y The y position of the coordinate.
 * @param new_state
 */
void Surface::set_state(int x, int y, int new_state) {
    surface[y][x] *= new_state;
}

/**
 * Calculates the current energy of the neighbours
 * located adjacent to the given coordinates.
 * @param x The horizontal coordinate of the vertex.
 * @param y The vertical coordinate of the vertex.
 * @return The total energy of the adjacent vertices.
 */
int Surface::energy_neighbours(int x, int y) {
    int eN = 0;
    eN += get_state((x + 1) % size, y);
    eN += get_state((x - 1), y);
    eN += get_state(x, (y + 1) % size);
    eN += get_state(x, (y - 1));
    return eN;
}

/**
 * Calculates the total energy of the surface.
 * @return The total energy of the surface.
 */
int Surface::calculate_energy() {
    int totalEnergy = 0;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            totalEnergy += Surface::get_state(x, y) * -(Surface::energy_neighbours(x, y));
        }
    }
    return totalEnergy;
}

/**
 * Calculates the total magnetism of the surface.
 * @return The total magnetism of the surface.
 */
int Surface::calculate_magnetism() {
    int totalMagnetism = 0;
    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            totalMagnetism += Surface::get_state(x, y);
        }
    }
    return totalMagnetism;
}

/**
 * Generates a random point on the surface.
 * @return The generate random point on the surface.
 */
int Surface::random_point() const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, size-1);
    return dis(gen);
}

/**
 * Generates a random number between 0 and 1.
 * @return The random number.
 */
float random_real() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}

/**
 * Main calculations that are made to determine the spin state
 * of a position.
 * @param lattice The Surface class instance which hold the current
 *                  state.
 * @param coords The coordinates to check.
 */
void calculate_spin(Surface lattice, int coords[2]) {
    int state = lattice.get_state(coords[0], coords[1]);
    int neighbours = lattice.energy_neighbours(coords[0], coords[1]);

    int energy = 2 * state * neighbours;
    float random = random_real();
    double diff = -energy / lattice.beta * 10 * 23; // Adding back decimal place Boltzmann.
    double thresh = exp(diff);

    if (energy <= 0 || random < thresh) {
        lattice.set_state(coords[0], coords[1], -1);
    }
}

/**
 * Runs the simulation keeping track of various states.
 * @param lattice The lattice to simulate.
 * @return The exit state of the application.
 */
int simulate(Surface lattice) {
    lattice.save();
    for (int i = 0; i < lattice.loops; ++i) {
        lattice.avgEnergy[i] = lattice.calculate_energy();
        lattice.avgMag[i] = lattice.calculate_magnetism();

        for (int j = 0; j < lattice.size; ++j) {
            for (int k = 0; k < lattice.size; ++k) {
                int coords[2] = {j,k};
                calculate_spin(lattice, coords);
            }
        }
    }
    lattice.complete = true;
    lattice.save();
    return EXIT_SUCCESS;
}

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
