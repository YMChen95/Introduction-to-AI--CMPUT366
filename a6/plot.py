import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    V1 = open('result1.txt', "r")
    V2 = open('result2.txt', "r")
    V3 = open('result3.txt', "r")

    y1 = []
    y2 = []
    y3 = []
    for line in V1:
        line = line.strip()
        y1.append(line)

    for line in V2:
        line = line.strip()
        y2.append(line)

    for line in V3:
        line = line.strip()
        y3.append(line)

    plt.show()
    plt.plot(y1, label="tabular_agent")
    plt.plot(y2, label="tile_agent")
    plt.plot(y3, label="aggregation_agent")
    plt.xlim([0, 2000])

    plt.legend()
    plt.show()
