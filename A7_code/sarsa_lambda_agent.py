from utils import rand_in_range, rand_un
import numpy as np
import pickle
from tiles3 import tiles, IHT

Q = None
w = None
last_state = None
gamma = 1
iht = IHT(4096)

sizeTilings = None
alpha = 0.1/8
lam = 0.9
Value_func = None
z = None
last_action = None


def agent_init():
    global w, last_state, iht, Value_func, sizeTilings, Q, z, last_action
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    Q = np.zeros((8 ,8,3))
    w = np.random.uniform(-0.001, 0, 1944)
    z = np.zeros(1944)
    sizeTilings = np.array([8, 8])
    last_state = np.zeros(2)
    last_action = np.zeros(1)


def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global last_state, state1, last_action

    x = 8 * state[0] / (0.5 + 1.2)
    xdot = 8 * state[1] / (0.07 + 0.07)
    current_state = [x, xdot]

    feature1 = np.zeros(1944)
    feature_list1 = tiles(iht, 8, current_state, [0])

    feature2 = np.zeros(1944)
    feature_list2 = tiles(iht, 8, current_state, [1])

    feature3 = np.zeros(1944)
    feature_list3 = tiles(iht, 8, current_state, [2])

    for i in feature_list1:
        feature1[i] = 1
    for i in feature_list2:
        feature2[i] = 1
    for i in feature_list3:
        feature3[i] = 1

    v1 = np.dot(w, feature1)
    v2 = np.dot(w, feature2)
    v3 = np.dot(w, feature3)

    action = np.argmax([v1, v2, v3])
    last_action = action
    last_state = current_state

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: floating point
    """

    global last_state, w, Value_func, z, last_action

    delta = reward

    x = 8 * state[0] / (0.5 + 1.2)
    xdot = 8 * state[1] / (0.07 + 0.07)
    current_state = [x, xdot]

    last_indices = tiles(iht, 8, last_state, [last_action])
    for i in last_indices:
        delta -= w[i]
        z[i] = 1

    feature1 = np.zeros(1944)
    feature_list1 = tiles(iht, 8, current_state, [0])

    feature2 = np.zeros(1944)
    feature_list2 = tiles(iht, 8, current_state, [1])

    feature3 = np.zeros(1944)
    feature_list3 = tiles(iht, 8, current_state, [2])

    for i in feature_list1:
        feature1[i] = 1
    for i in feature_list2:
        feature2[i] = 1
    for i in feature_list3:
        feature3[i] = 1

    v1 = np.dot(w, feature1)
    v2 = np.dot(w, feature2)
    v3 = np.dot(w, feature3)

    action = np.argmax([v1, v2, v3])
    indices = tiles(iht, 8, current_state, [action])
    for i in indices:
        delta += w[i]

    w += alpha * z * delta
    z = z * gamma * lam
    last_action = action
    last_state = current_state

    return action


def agent_end(reward):

    global w, Value_func
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    delta = reward
    last_indices = tiles(iht, 8, last_state, [last_action])
    for i in last_indices:
        delta -= w[i]
        z[i] = 1
    w += alpha * z * delta


    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global Value_func, w
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):

        return [w, iht]
    else:
        return "I don't know what to return!!"