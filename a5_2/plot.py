#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":


    V1 = open('part2_result.txt',"r")
    alpha = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
    listy =[]
    for line in V1:
        #x1 = int(line.split(" ")[0])
        y1 = int(line.split(" ")[0])

        listy.append(y1)

    plt.plot(alpha,listy)
    plt.ylim([20,80])
    plt.yticks([20,40,60,80])
    plt.xlim([0,1])
    #plt.xticks([0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0])
    plt.xlabel('Alpha')
    plt.ylabel('Steps per Episodes')
    plt.legend( loc="best")
    plt.show()
