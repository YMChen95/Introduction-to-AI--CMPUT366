#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
		   for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import rand_in_range, rand_un
import numpy as np
import pickle

epsilon = 0.1
alpha_step = 0.5


Q = None
last_action = None
last_state = None

def agent_init():
    global Q, actions, last_action, last_state, total_actions
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    # initialize the policy array in a smart way
    Q = np.zeros((10, 7, total_actions))
    last_state = np.zeros(2)
    last_action = 0

    return


def agent_start(state):
    global Q, actions, last_action, last_state, total_actions
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    select_option = np.array([0, 1])
    option = np.random.choice(select_option, p=[epsilon, 1 - epsilon])

    if option ==0:
        action = rand_in_range(total_actions)
    else:
        action = np.argmax(Q[state[0]][state[1]])

    last_action = action
    last_state = state

    return action


def agent_step(reward, state):  # returns NumPy array, reward: floating point, this_observation: NumPy array
    global Q, last_action, last_state, total_actions
    """
    Arguments: reward: floting point, state: integer
    Returns: action: floating point
    """
    # select an action, based on Q

    select_option = np.array([0, 1])
    option = np.random.choice(select_option, p=[epsilon, 1 - epsilon])

    current_x = state[0]
    current_y = state[1]
    last_x =last_state[0]
    last_y = last_state[1]
    if option == 0:
        action = rand_in_range(total_actions)
    else:
        action = np.argmax(Q[state[0]][state[1]])

    Q[last_x][last_y][last_action] += alpha_step * (reward + Q[current_x][current_y][action] - Q[last_x][last_y][last_action])

    last_action = action
    last_x = current_x
    last_y = current_y
    last_state =[last_x, last_y]
    return action


def agent_end(reward):
    global Q, last_action, last_state

    Q[last_state[0]][last_state[1]][last_action] += alpha_step * (reward - Q[last_state[0]][last_state[1]][last_action])

    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up

    return


def agent_message(in_message):  # returns string, in_message: string
    global Q, total_actions
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return
    elif str(in_message) == "0":
        total_actions = 8
    elif str(in_message) == "1":
        total_actions = 9
    else:
        return "I don't know what to return!!"

