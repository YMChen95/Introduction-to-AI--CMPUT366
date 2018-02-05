from utils import rand_in_range, rand_un
import numpy as np
import pickle

Q = None
epsilon = 0.1
alpha = 0.1
gamma = 0.95
model = {}
last_action = None
last_state = None


def agent_init():
    global Q, last_action, last_state, model

    Q = np.zeros((6, 9, 4))
    last_action = np.zeros(1)
    last_state = np.zeros(2)
    for i in range(6):
        for j in range(9):
            model[(i, j)] = {0:[0, 0, -1], 1:[0, 0, -1], 2:[0, 0, -1], 3:[0, 0, -1]}


def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    global Q, last_action, epsilon, last_state

    select_option = np.array([0, 1])
    option = np.random.choice(select_option, p=[epsilon, 1 - epsilon])
    x = state[0]
    y = state[1]

    if option == 0:
        action_num = rand_in_range(4)

    else:
        action_num = np.argmax(Q[y][x])
        if Q[y][x][action_num] == 0:
            action_num = rand_in_range(4)

    last_action[0] = action_num
    action = last_action[0]
    last_state = state

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: floating point
    """

    global Q, last_action, epsilon, alpha, last_state, gamma, n

    select_option = np.array([0, 1])
    option = np.random.choice(select_option, p=[epsilon, 1 - epsilon])
    current_x = state[0]
    current_y = state[1]

    if option == 0:
        action_num = rand_in_range(4)  # change this to 9 to rand in 9 actions

    else:
        action_num = np.argmax(Q[current_y][current_x])
        if Q[current_y][current_x][action_num] == 0:
            action_num = rand_in_range(4)

    Q[last_state[1]][last_state[0]][last_action[0]] += alpha*(reward+gamma*np.max(Q[current_y][current_x]) - Q[last_state[1]][last_state[0]][last_action[0]])

    model[(last_state[1], last_state[0])][last_action[0]] = [current_x, current_y, reward]

    for i in range(n):
        exist = False
        while not exist:
            model_x = rand_in_range(9)
            model_y = rand_in_range(6)
            model_action = rand_in_range(4)
            if model[(model_y, model_x)][model_action][2] != -1:
                exist = True

        S_x = model[(model_y, model_x)][model_action][0]
        S_y = model[(model_y, model_x)][model_action][1]
        Rwd = model[(model_y, model_x)][model_action][2]

        Q[model_y][model_x][model_action] += alpha * (Rwd + gamma * np.max(Q[S_y][S_x])- Q[model_y][model_x][model_action])


    last_action[0] = action_num
    last_state = state
    action = last_action[0]

    return action


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    global Q, last_action, alpha, last_state

    Q[last_state[1]][last_state[0]][last_action[0]] += alpha * (reward - Q[last_state[1]][last_state[0]][last_action[0]])

    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    global Q, n
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return pickle.dumps(np.max(Q, axis=1), protocol=0)
    elif (in_message == '0'):
        n = 0
    elif (in_message == '5'):
        n = 5
    elif (in_message == '50'):
        n = 50
    else:
        return "I don't know what to return!!"