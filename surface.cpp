#include "surface.h"

using namespace std;

/**
 * Constructor method takes the size and temperature
 * of the surface. It then creates the N^2 surface.
 * @param size The size of the N^2 lattice.
 * @param temp  The temperature of the lattice.
 */
Surface::Surface(int loops, int size, double temp) {
    Surface::size = size;
    Surface::temp = temp;
    Surface::loops = loops;
    surface = create_surface();
    beta = boltzmann * temp;
    avgEnergy = new int[loops];
    avgMag = new int[loops];
}

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
float Surface::random_real() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}

/**
 * Calculations that are made to determine the spin state
 * of a position.
 * @param lattice The Surface class instance which hold the current
 *                  state.
 * @param coords The coordinates to check.
 */
void Surface::calculate_spin(int coords[2]) {
    int state = get_state(coords[0], coords[1]);
    int neighbours = energy_neighbours(coords[0], coords[1]);

    int energy = 2 * state * neighbours;
    float random = random_real();
    double diff = -energy / beta * 10 * 23; // Adding back decimal place Boltzmann.
    double thresh = exp(diff);

    if (energy <= 0 || random < thresh) {
        set_state(coords[0], coords[1], -1);
    }
}


