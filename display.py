import matplotlib.pyplot as plot
import numpy

if __name__ == '__main__':

    file = open('test.out', 'r')
    lines = file.readlines()

    for line in lines:
        data = numpy.matrix(line)

        plot.matshow(data)
        plot.show()