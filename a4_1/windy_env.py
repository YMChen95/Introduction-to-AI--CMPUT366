#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np


current_state = None
wind_strength = [0,0,0,1,1,1,2,2,1,0]


def env_init():
    global current_state
    current_state = np.zeros(2)



def env_start():
    """ returns numpy array """
    global current_state

    current_state = [0,3]
    return current_state


def env_step(action):
    '''
    parameter: action [int,int]
    return: result ={"reward": int , "state": numpy array, "isTerminal": boolean}
    '''
    global current_state

    action_direct = None
    is_terminal = False

    if action == 0:
        action_direct = [0, 1]
    elif action == 1:
        action_direct = [0, -1]
    elif action == 2:
        action_direct = [-1, 0]
    elif action == 3:
        action_direct = [1, 0]
    elif action == 4:
        action_direct = [-1, 1]
    elif action == 5:
        action_direct = [1, 1]
    elif action == 6:
        action_direct = [-1, -1]
    elif action == 7:
        action_direct = [1, -1]
    elif action == 8:
        action_direct = [0, 0]

    else:
        print "Invalid action taken!!"
        exit(1)
    current_x = current_state[0]
    current_y = current_state[1]

    new_x = current_x + action_direct[0]
    new_y = current_y + action_direct[1]

    if new_x >= 0 and new_y >= 0 and new_x <= 9 and new_y <= 6:
        current_state = [new_x, new_y]



    if current_state[1] + wind_strength[current_x] > 6:
        current_state[1] = 6
    else:
        current_state[1] += wind_strength[current_x]

    if current_state == [7,3]:
        is_terminal= True

    if is_terminal is True:
        reward = 1
    else:
        reward = -1

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