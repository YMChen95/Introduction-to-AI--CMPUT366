import numpy as np
from rl_glue import *  # Required for RL-Glue


def tabular(true_value):
    RLGlue("random_walk_env", "tabular_agent")

    error = np.zeros(5000)

    for run in range(10):
        RL_init()
        for episode in range(2000):
            print('agent1, run:', run, 'episode:', episode)
            RL_episode(10000)
            app_value = RL_agent_message("ValueFunction")
            error[episode] += np.sqrt(np.mean(np.power(true_value - app_value, 2)))

        RL_cleanup()


    file1 = open("result1.txt", "w")
    for i in range(2000):
        file1.write("{0}\n".format(error[i]/10))
    file1.close()

def tile(true_value):
    RLGlue("random_walk_env", "tile_agent")

    error = np.zeros(5000)

    for run in range(10):
        RL_init()
        for episode in range(2000):
            print('agent2, run:', run, 'episode:', episode)
            RL_episode(10000)
            app_value = RL_agent_message("ValueFunction")
            error[episode] += np.sqrt(np.mean(np.power(true_value - app_value, 2)))

        RL_cleanup()


    file2 = open("result2.txt", "w")
    for i in range(2000):
        file2.write("{0}\n".format(error[i]/10))
    file2.close()

def aggregation(true_value):
    RLGlue("random_walk_env", "aggregation_agent")

    error = np.zeros(5000)

    for run in range(10):
        RL_init()
        for episode in range(2000):
            print('agent3, run:', run, 'episode:', episode)
            RL_episode(10000)
            app_value = RL_agent_message("ValueFunction")
            error[episode] += np.sqrt(np.mean(np.power(true_value - app_value, 2)))
        RL_cleanup()


    file3 = open("result3.txt", "w")
    for i in range(2000):
        file3.write("{0}\n".format(error[i]/10))
    file3.close()



def compute_value_function():
    """
    Computes the value function for the 1000 state random walk as described in Sutton and Barto (2017).
    :return: The value function for states 1 to 1000. Index 0 is not used in this array (i.e. should remain 0).
    """
    state_prob = 0.5 / 100.0
    gamma = 1
    theta = 0.000001

    V = np.zeros(1001)

    delta = np.infty
    i = 0
    while delta > theta:
        i += 1
        delta = 0.0
        for s in range(1, 1001):
            v = V[s]
            value_sum = 0.0
            for transition in range(1, 101):
                right = s + transition
                right_reward = 0
                if right > 1000:
                    right_reward = 1
                    right = 0

                left = s - transition
                left_reward = 0
                if left < 1:
                    left_reward = -1
                    left = 0

                value_sum += state_prob * ((right_reward + gamma * V[right]) + (left_reward + gamma * V[left]))

            V[s] = value_sum
            delta = max(delta, np.abs(v - V[s]))

        print('true value:', i)

    return V

def main():

    np.random.seed(50)
    v_bar = compute_value_function()[1:]


    tabular(v_bar)
    tile(v_bar)
    aggregation(v_bar)
    print('done, use python plot.py to generate plot')

main()

