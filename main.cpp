#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>
#include <iostream>
#include <fstream>

using namespace std;

class Surface {
    public:
        bool out = false;
        char* outName = nullptr;

        /**
         * Constructor method takes the size and temperature
         * of the surface. It then creates the N^2 surface.
         * @param size The size of the N^2 lattice.
         * @param temp  The temperature of the lattice.
         */
        Surface(int size, int temp) {
            Surface::size = size;
            Surface::temp = temp;
            surface = create_surface();
        }

        void output_config(ostream &output) const;
        void save();
        int calculate_magnetism();
        int calculate_energy();

    private:
        int** surface;
        int size;
        int temp;

        int** create_surface() const;
        void output_surface(ostream &output) const;
        int state(int x, int y);
        int energy_neighbours(int x, int y);
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
* Uses Bernoulli distribution to determine the state
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
    file.open(outName);
    output_surface(file);
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
 * Finds the current state of the vertex in
 * that position.
 * @param x The horizontal coordinate of the vertex.
 * @param y The vertical coordinate of the vertex.
 * @return The current state of the vertex.
 */
int Surface::state(int x, int y) {
    if (x < 0) {
        x = size + x;
    } else if (y < 0) {
        y = size + y;
    }
    return surface[y][x];
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
    eN += state((x + 1) % size, y);
    eN += state((x - 1) % size, y);
    eN += state(x, (y + 1) % size);
    eN += state(x, (y - 1) % size);
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
            totalEnergy += Surface::state(x, y) * -(Surface::energy_neighbours(x, y));
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
            totalMagnetism += Surface::state(x, y);
        }
    }
    return totalMagnetism;
}

int simulate(long int n, Surface lattice) {
    lattice.save();
    for (int i = 0; i < n; ++i) {
        cout << "Energy: " << lattice.calculate_energy() << " Mag: :" << lattice.calculate_magnetism() << endl;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        std::cout << "Usage: ./ising n size temperature {output}" << std::endl;
        return 1;
    }

    long int n = strtol(argv[1], 0, 10);
    if (n <= 0) {
        std::cout << "'n' has to be a positive integer" << std::endl;
        return 1;
    }

    Surface lattice(atoi(argv[2]), atoi(argv[3]));

    if (argc == 5) {
        lattice.out = true;
        lattice.outName = argv[4];
    }
    lattice.output_config(std::cout);

    return simulate(n, lattice);
}
