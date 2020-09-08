import matplotlib.pyplot as plt
import numpy

if __name__ == '__main__':
    file = open('test.out', 'r')
    lines = file.readlines()

    i = 0
    for line in lines:
        if i < 2:
            data = numpy.matrix(line)
            plt.matshow(data)
        else:
            line = line[:-1]
            newline = line.split(' ')
            plt.plot(newline)
        i += 1
        plt.title(i)
        plt.show()
