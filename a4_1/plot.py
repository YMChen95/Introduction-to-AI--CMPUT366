#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":


    V1 = open('result.txt',"r")
    listx =[]
    listy =[]
    for line in V1:
        x = int(line.split(" ")[0])
        y = int(line.split(" ")[1])
        listx.append(x)
        listy.append(y)
    plt.plot(listy,listx)
    plt.xlim([0,8000])
    plt.xticks([0,1000,2000,3000,4000,5000,6000,7000,8000])
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.legend( loc="best")
    plt.show()
