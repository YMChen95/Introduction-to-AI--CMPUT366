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
    result= np.zeros(6)
    alpha = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]

    np.random.seed(23)
    for i in range(6):
        print("alpha = ",alpha[i])
        for run in range(10):
            RL_init()
            RL_agent_message(str(alpha[i]))
            num_episodes = 0
            while num_episodes < 50:
                RL_episode(1500)
                steps = RL_num_steps()
                result[i] += steps
                num_episodes += 1
            print('run :', run+1, 'done')

    RL_cleanup()
    #print(result)

    file = open("part2_result.txt","w")
    for i in range(len(result)):
        file.write("%d\n"%(float(result[i]/10/50)))
    file.close()

    print("Done, please use python plot.py to get plot\n")

