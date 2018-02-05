#!/usr/bin/env python

"""
	Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
	Jacobsen, Victor Silva, Sina Ghiassian
	Purpose: Implementation of the interaction between the Gambler's problem environment
	and the Monte Carlon agent using RL_glue.
	For use in the Reinforcement Learning course, Fall 2017, University of Alberta
"""
import numpy as np
import pickle
import time
from rl_glue import *  # Required for RL-Glue
RLGlue("dyna_env", "agent")




start_time = time.time()
if __name__ == "__main__":

    max_steps = 50
    steps = 0
    result= np.zeros((3,51))
    n = [0,5,50]

    np.random.seed(23)
    for i in range(3):
        print("n = ",n[i])
        for run in range(10):
            RL_init()
            RL_agent_message(str(n[i]))
            num_episodes = 0
            while num_episodes < 51:
                RL_episode(1500)
                steps = RL_num_steps()
                result[i][num_episodes] += steps
                num_episodes += 1
            print('run :', run+1, 'done')

    RL_cleanup()
    #print(result)

    file1 = open("result1.txt","w")
    for i in range(len(result[1])):
        if i >0:
            file1.write("%d %d\n"%(i, result[0][i]/10))
    file1.close()

    file2 = open("result2.txt","w")
    for i in range(len(result[2])):
        if i >0:
            file2.write("%d %d\n"%(i, result[1][i]/10))
    file2.close()

    file3 = open("result3.txt","w")
    for i in range(len(result[2])):
        if i >0:
            file3.write("%d %d\n"%(i, result[2][i]/10))
    file3.close()

    print("Done, please use python plot.py to get plot\n")

