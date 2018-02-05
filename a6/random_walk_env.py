#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np


tiling = None
#block = [[2,2],[2,3],[2,4],[5,1],[7,3],[7,4],[7,5]]
#wind_strength = [0,0,0,1,1,1,2,2,1,0]


def env_init():
    global current_state
    current_state = np.zeros(1)



def env_start():
    """ returns numpy array """
    global current_state

    current_state[0] = 500

    return current_state


def env_step(action):
    '''
    parameter: action [int,int]
    return: result ={"reward": int , "state": numpy array, "isTerminal": boolean}
    '''
    global current_state

    step = np.random.randint(1, 100 + 1)


    if action == 1:
        direction = 1
    else:
        direction = -1

    true_action = direction * step

    current_state[0] += true_action



    if current_state[0] > 1000:
        is_terminal= True
        reward = 1

    elif current_state[0] < 1:
        is_terminal = True
        reward = -1

    else:
        is_terminal = False
        reward = 0



    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}

    return result

def env_cleanup():
    #
    return


def env_message(in_message):  # returns string, in_message: string
    """
    Arguments
    ---------
    inMessage : string
        the message being passed
    Returns
    -------
    string : the response to the message
    """
    return ""