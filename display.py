import matplotlib.pyplot as plt
import numpy
import sys

if __name__ == '__main__':
    file = open(sys.argv[1], 'r')
    lines = file.readlines()
    titles = {1: "Starting State Matrix", 2: "Final State Matrix", 3: "Energy vs Time", 4: "Magnetism vs Time"}
    labels = {3: "Energy", 4: "Magnetism"}
    i = 0
    for line in lines:
        i += 1
        if i < 3:
            data = numpy.matrix(line)
            plt.matshow(data)
            plt.xlabel("x-axis")
            plt.ylabel("y-axis")
        else:
            line = line[:-1]
            newline = line.split(' ')
            newline = numpy.array(newline).astype(numpy.float)
            plt.plot(newline)
            plt.xlabel("Time")
            plt.ylim(numpy.min(newline), numpy.max(newline))
            plt.ylabel(labels[i])
        plt.title(titles[i])
        plt.show()
