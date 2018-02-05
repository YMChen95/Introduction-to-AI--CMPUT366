from utils import rand_in_range, rand_un
import time as sys
import numpy as np

pre_obs_state = [[1,3],[2,3],[3,3]]
pre_obs_action = {}
state = [1,3]
action = 2
action1 = 0
action2 = 1

new_state = tuple(state)

pre_obs_action[new_state] = []
if action not in pre_obs_action[tuple(state)]:
    pre_obs_action[tuple(state)].append(action)
pre_obs_action[new_state].append(action1)
pre_obs_action[new_state].append(action2)

S_x = pre_obs_state[rand_in_range(len(pre_obs_state))][0]
S_y = pre_obs_state[rand_in_range(len(pre_obs_state))][1]


if state not in pre_obs_state:
    pre_obs_state.append(state)

print((S_x,S_y))




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
alpha = 0.1
gamma = 0.95
n = 50

Q = None
model = None
last_action = None
last_state = None
pre_obs_state =[]
pre_obs_action = {}



def agent_init():
    global Q, last_action, last_state, model, n
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    Q = np.zeros((6, 9, 4))  # change it to 9 for 9 actions
    last_action = 0
    last_state = np.zeros(2)
    model = np.full((6,9,4,3), -1)


def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global Q, last_action, epsilon, last_state, pre_obs_state,pre_obs_action

    select_option = np.array([0, 1])
    option = np.random.choice(select_option, p=[epsilon, 1 - epsilon])
    x = state[0]
    y = state[1]

    if option == 0:
        action = rand_in_range(4)

    else:
        action = np.argmax(Q[y][x])
        if Q[y][x][action] == 0:
            action = rand_in_range(4)

    pre_obs_action[tuple(state)] = []
    pre_obs_action[tuple(state)].append(action)

    last_action = action
    last_state = state

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: floating point
    """

    global Q, last_action, epsilon, alpha, last_state, gamma, pre_obs_state,pre_obs_action

    select_option = np.array([0, 1])
    option = np.random.choice(select_option, p=[epsilon, 1 - epsilon])
    x = state[0]
    y = state[1]

    if option == 0:
        action = rand_in_range(4)  # change this to 9 to rand in 9 actions

    else:
        action = np.argmax(Q[y][x])
        if Q[y][x][action] == 0:
            action = rand_in_range(4)

    if state not in pre_obs_state:
        pre_obs_state.append(state)
        pre_obs_action[tuple(state)] = []
        if action not in pre_obs_action[tuple(state)]:
            pre_obs_action[tuple(state)].append(action)

    Q[last_state[1]][last_state[0]][last_action] += alpha*(reward+gamma*np.max(Q[y][x]) - Q[last_state[1]][last_state[0]][last_action])

    model[last_state[1]][last_state[0]][last_action] = [x, y, reward]

    for i in range(n):
        rand_index = rand_in_range(len(pre_obs_state))
        S_x = pre_obs_state[rand_index][0]
        S_y = pre_obs_state[rand_index][1]

        index = rand_in_range(len(pre_obs_action[(S_x, S_y)]))
        rand_action = pre_obs_action[(S_x,S_y)][index]

        next_state = [model[S_y][S_x][rand_action][0],model[S_y][S_x][rand_action][1]]
        Rwd = model[S_y][S_x][rand_action][2]

        Q[S_y][S_x][rand_action] += alpha * (Rwd + gamma * np.max(Q[next_state[1]][next_state[0]])- Q[S_y][S_x][rand_action])


    last_action = action
    last_state = state

    return action


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    global Q, last_action, alpha, last_state

    Q[last_state[1]][last_state[0]][last_action] += alpha * (reward - Q[last_state[1]][last_state[0]][last_action])

    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global Q
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return pickle.dumps(np.max(Q, axis=1), protocol=0)
    else:
        return "I don't know what to return!!"


