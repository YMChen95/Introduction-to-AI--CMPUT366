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
RLGlue("windy_env", "agent")




start_time = time.time()
if __name__ == "__main__":

    while True:
        try:
            user_input = int(raw_input("Please enter 0 for 8 actions, Enter 1 for 9 actions:"))
        except ValueError:
            print("\ninvalid input, try again:")
            continue
        if user_input != 1 and user_input != 0:
            print("invalid input, try again:\n")
        else:
            break

    max_steps = 8000
    total_steps = 0
    result=[]

    RL_agent_message(user_input)
    RL_init()
    while total_steps < max_steps:
        RL_episode(10000)
        total_steps += RL_num_steps()
        result.append(total_steps)

    RL_cleanup()
    #print(result)
    print("Done, please use python plot.py to get plot\n")

    file = open("result.txt","w")
    for i in range(len(result)):
        file.write("%d %d\n"%(i+1, result[i]))
    file.close()


