# Monte Carlo Simulation of Ising Model
The simulation can either be run manually or by using the **test.sh** script.
This will initiate the simulation with a pre-defined simulation. The simulation's iterations *n*, size *N* and temperatures *temp* can be
changed by updating the script.

To run the simulation manually you can use **make** to create the file *./ising*.

This can then be using to run a simulation like so "*./ising 5 200 2.2 output*" which will output the information to a 
file called output. The first two rows will contain space delimitered matrix data of the lattice state. The next two rows
will contain the energy and magnetism over-time at the end of each iterative cycle. By running **display.py** you can visualise the data.