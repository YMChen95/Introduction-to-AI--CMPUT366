#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle

alpha = 0.5
w = None
x = None
last_state2 = None


def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """

    #initialize the policy array in a smart way
    global w, x, last_state
    
    last_state = 0
    w = np.zeros(1000)
    x = np.identity(1000)
    
    
            
def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts 
    
    global w, x, last_state
    
    action = np.random.binomial(1, 0.5)
    last_state = state[0]

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global w, x, last_state
    
    current_state = state[0]
    w = w + alpha*(reward + w[current_state -1]- w[last_state-1])* x[last_state-1]
    last_state = current_state
    action = np.random.binomial(1, 0.5)
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    global w, x, last_state

    w += alpha * (reward - w[last_state-1])* x[last_state-1]
    
    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global w
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return w
    else:
        return "I don't know what to return!!"

