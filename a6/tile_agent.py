#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
		   for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""
import numpy as np


from importlib import import_module

tile = import_module("tiles3")
iht = tile.IHT(3000)
alpha = 0.01

v = None
w = None
tiling = None
tiling_state = None
last_state2 = None





def agent_init():
    global w,v,tiling,tiling_state,last_state2
    v = np.zeros(1000)
    w = np.zeros(1200)
    tiling = np.zeros(1)
    tiling_state = np.zeros(1)
    last_state2 = np.zeros(1)


    return

def agent_start(state):
    global w,tiling,tiling_state,last_state2

    tiling[0] = float(state[0] / 100.0)
    tiling_state[0] = tiling[0]
    last_state2[0] = state[0]
    action = np.random.binomial(1, 0.5)




    return action


def agent_step(reward, state):
    global w,tiling,tiling_state,v,last_state2

    state1 = np.zeros(1200)
    state2 = np.zeros(1200)

    tiling[0] = float(state[0] / 200.0)
    currentx = tile.tiles(iht, 50, tiling)
    pre_x =  tile.tiles(iht, 50, tiling_state)

    action = np.random.binomial(1, 0.5)
    for index in currentx:
        state1[index] = 1
    for index in pre_x:
        state2[index] = 1

    w += alpha * (reward + np.dot(w, state1) - np.dot(w, state2)) * state2
    v[last_state2[0] - 1] = np.dot(w, state2)

    tiling_state[0] = tiling[0]
    last_state2[0] = state[0]


    return action


def agent_end(reward):
    global w,tiling,tiling_state,v,last_state2

    state2 = np.zeros(1200)
    lastx =  tile.tiles(iht, 50, tiling_state)

    for index in lastx:
        state2[index] = 1


    w += alpha*(reward- np.dot(w,state2))*state2
    v[last_state2[0] - 1] = np.dot(w, state2)

    return

def agent_cleanup():
    """
	This function is not used
	"""
    # clean up

    return

def agent_message(in_message): # returns string, in_message: string
    global v
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return v
    else:
        return "I don't know what to return!!"

