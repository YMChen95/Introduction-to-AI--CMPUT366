#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Mohammad M. Ajallooeian, Sina Ghiassian
  Last Modified on: 21/11/2017

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("mountaincar", "sarsa_lambda_agent")

from tiles3 import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

if __name__ == "__main__":
    num_episodes = 1000
    num_runs = 1

    #steps = np.zeros([num_runs,num_episodes])

    for r in range(num_runs):
        print "run number : ", r
        RL_init()
        for e in range(num_episodes):
            # print '\tepisode {}'.format(e+1)
            RL_episode(0)
      
    fout = open('value', 'w')
    steps = 50
    
    x = np.arange(-1.2, 0.5, 1.7 / steps)
    y = np.arange(-0.07, 0.07, 0.14 / steps)
    Q = np.zeros([steps, steps])
    x, y = np.meshgrid(x, y)
         
    
    [w, iht] = RL_agent_message("ValueFunction")
    for i in range(steps):
        p = -1.2 + (i * 1.7 / steps)
        for j in range(steps):
            v = -0.07 + (j * 0.14 / steps)
            values = []
            
            for action in range(3):
                X = np.zeros_like(w)
                for t in tiles(iht, 8, [8*p/(0.5+1.2), 8*v/(0.07+0.07)], [action]):
                    X[t] = 1.0
                values.append(-w.dot(X))
                
            height = np.amax(values)
            Q[j][i] = height
            fout.write(repr(height)+'')
        fout.write('\n')
    fout.close()

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')  
    ax.set_xticks([-1.2, 0.5])
    ax.set_yticks([-0.07, 0.07])
    ax.set_zticks([0, np.amax(Q)])
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost to go')
    ax.plot_surface(x, y, Q,cmap=cm.coolwarm)
    plt.show()
