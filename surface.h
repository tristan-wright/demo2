#include <iostream>
#include <cstdlib>
#include <random>
#include <fstream>

#ifndef ISING_SURFACE_H
#define ISING_SURFACE_H

using namespace std;

class Surface {
public:
    Surface(char* name, int loops, int size, double temp);
    void save();
    void clear();
    int calculate_magnetism();
    int calculate_energy();
    void calculate_spin(int coords[2]);
    void output_surface(ostream &output) const;

    // Used to debug surface configuration
    void output_config(ostream &output) const;

    bool complete = false;
    bool out = false;
    char* outName = nullptr;
    double beta;
    int size;
    int* avgEnergy;
    int* avgMag;
    int loops;
    char* name;
    int** surface;
    const double boltzmann = 1.3806503;
    double temp;

private:
    int random_point() const;
    int energy_neighbours(int x, int y);
    int get_state(int x, int y);
    void set_state(int x, int y, int new_state);
    void output_eng_mag(ostream &output) const;
    int** create_surface() const;
    float random_real();
};

#endif //ISING_SURFACE_H
