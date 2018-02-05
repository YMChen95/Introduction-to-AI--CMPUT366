import os
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

filename = 'costToGo.npy'

if os.path.exists(filename):
    Z = np.load(filename)
    numTilings = 8
    pos_max = 0.5
    pos_min = -1.2
    vel_max = 0.07
    vel_min = -0.07
    position = np.arange(pos_min, pos_max, (pos_max - pos_min) / 50.0)
    velocity = np.arange(vel_min, vel_max, (vel_max - vel_min) / 50.0)
    X = np.zeros((50))
    Y = np.zeros((50))

    X, Y = np.meshgrid(position, velocity)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z)

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('cost to go')

    plt.show()