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

        void output_config(ostream &output);
        void save();
        int get_neighbours(int x, int y);

    private:
        int** surface;
        int size;
        int temp;

        int** create_surface() const;
        void output_surface(ostream &output) const;
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
void Surface::output_config(ostream& output) {
    output << size << ',' << temp << std::endl;
}



int simulate(long int n, Surface lattice) {
    for (int i = 0; i < n; ++i) {
        lattice.save();
    }
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
