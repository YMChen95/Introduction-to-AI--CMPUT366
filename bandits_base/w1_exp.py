#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian
  Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
  Last Modified by: Andrew Jacobsen, Victor Silva, Mohammad M. Ajallooeian
  Last Modified on: 16/9/2017

  Experiment runs 2000 runs, each 1000 steps, of an n-armed bandit problem
"""

from rl_glue import *  # Required for RL-Glue
RLGlue("w1_env", "w1_agent")

import numpy as np
import sys

def save_results(data, data_size, filename): # data: floating point, data_size: integer, filename: string
    with open(filename, "w") as data_file:
        for i in range(data_size):
            data_file.write("{0}\n".format(data[i]))

def getOptimalAction():

    return int(RL_env_message("get optimal action"))

def chooseselection(message):
    RL_agent_message(message)
    return None

if __name__ == "__main__":
    num_runs = 2000
    max_steps = 1000

    # array to store the results of each step
    optimal_action = np.zeros(max_steps)


    while True:
        try:
            user_input = int(raw_input("Enter 0 for epsilon=0 Q1=5, Enter 1 for epsilon=0.1 Q1=0:"))
        except ValueError:
            print("\ninvalid input, try again:")
            continue
        if user_input != 1 and user_input != 0:
            print("invalid input, try again:\n")
        else:
            break


    print "\nPrinting one dot for every run: {0} total Runs to complete".format(num_runs)
    chooseselection(user_input)
    for k in range(num_runs):

        RL_init()
        best_action = getOptimalAction()
        RL_start()

        for i in range(max_steps):

            # RL_step returns (reward, state, action, is_terminal); we need only the
            # action in this problem
            action = RL_step()[2]
        
            if action[0] == best_action:
                optimal_action[i] += 1

        RL_cleanup()
        print ".",
        sys.stdout.flush()

    save_results(optimal_action / num_runs, max_steps, "RL_EXP_OUT.dat")
    print "\nDone"
