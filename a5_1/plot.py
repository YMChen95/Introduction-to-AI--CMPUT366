#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":


    V1 = open('result1.txt',"r")
    listx1 =[]
    listy1 =[]
    for line in V1:
        x1 = int(line.split(" ")[0])
        y1 = int(line.split(" ")[1])
        listx1.append(x1)
        listy1.append(y1)
    V1 = open('result2.txt',"r")
    listx2 =[]
    listy2 =[]
    for line in V1:
        x2 = int(line.split(" ")[0])
        y2 = int(line.split(" ")[1])
        listx2.append(x2)
        listy2.append(y2)
    V1 = open('result3.txt',"r")
    listx3 =[]
    listy3 =[]
    for line in V1:
        x3 = int(line.split(" ")[0])
        y3 = int(line.split(" ")[1])
        listx3.append(x3)
        listy3.append(y3)
    plt.plot(listx1,listy1)
    plt.plot(listx2,listy2)
    plt.plot(listx3,listy3)
    plt.xlim([0,50])
    plt.xticks([2,10,20,30,40,50])
    plt.xlabel('Time steps')
    plt.ylabel('Steps per Episodes')
    plt.legend( loc="best")
    plt.show()
