import numpy as np
import matplotlib.pyplot as plt

N_STATES = 1000
states = np.arange(1, N_STATES + 1)
trueStateValues = np.arange(-1001, 1003, 2) / 1001.0
START_STATE = 500
END_STATES = [0, N_STATES + 1]
ACTION_LEFT = -1
ACTION_RIGHT = 1
ACTIONS = [ACTION_LEFT, ACTION_RIGHT]
STEP_RANGE = 100


def figure9_1():
    nEpisodes = int(1e5)
    alpha = 2e-5

    # we have 10 aggregations in this example, each has 100 states
    valueFunction = ValueFunction(10)
    distribution = np.zeros(N_STATES + 2)
    for episode in range(0, nEpisodes):
        print('episode:', episode)
        gradientMonteCarlo(valueFunction, alpha, distribution)

    distribution /= np.sum(distribution)
    stateValues = [valueFunction.value(i) for i in states]

    plt.figure(0)
    plt.plot(states, stateValues, label='Approximate MC value')
    plt.plot(states, trueStateValues[1: -1], label='True value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()

    plt.figure(1)
    plt.plot(states, distribution[1: -1], label='State distribution')
    plt.xlabel('State')
    plt.ylabel('Distribution')
    plt.legend()

class ValueFunction:
    # @numOfGroups: # of aggregations
    def __init__(self, numOfGroups):
        self.numOfGroups = numOfGroups
        self.groupSize = N_STATES // numOfGroups

        # thetas
        self.params = np.zeros(numOfGroups)

    # get the value of @state
    def value(self, state):
        if state in END_STATES:
            return 0
        groupIndex = (state - 1) // self.groupSize
        return self.params[groupIndex]

    # update parameters
    # @delta: step size * (target - old estimation)
    # @state: state of current sample
    def update(self, delta, state):
        groupIndex = (state - 1) // self.groupSize
        self.params[groupIndex] += delta

def gradientMonteCarlo(valueFunction, alpha, distribution=None):
    currentState = START_STATE
    trajectory = [currentState]

    # We assume gamma = 1, so return is just the same as the latest reward
    reward = 0.0
    while currentState not in END_STATES:
        action = getAction()
        newState, reward = takeAction(currentState, action)
        trajectory.append(newState)
        currentState = newState

    # Gradient update for each state in this trajectory
    for state in trajectory[:-1]:
        delta = alpha * (reward - valueFunction.value(state))
        valueFunction.update(delta, state)
        if distribution is not None:
            distribution[state] += 1

def takeAction(state, action):
    step = np.random.randint(1, STEP_RANGE + 1)
    step *= action
    state += step
    state = max(min(state, N_STATES + 1), 0)
    if state == 0:
        reward = -1
    elif state == N_STATES + 1:
        reward = 1
    else:
        reward = 0
    return state, reward

# get an action, following random policy
def getAction():
    if np.random.binomial(1, 0.5) == 1:
        return 1
    return -1



def figure9_10():

    # My machine can only afford one run, thus the curve isn't so smooth
    runs = 1

    # number of episodes
    episodes = 1000

    numOfTilings = 50

    # each tile will cover 200 states
    tileWidth = 200

    # how to put so many tilings
    tilingOffset = 4

    labels = ['tile coding (50 tilings)', 'state aggregation (one tiling)']

    # track errors for each episode
    errors = np.zeros((len(labels), episodes))
    for run in range(0, runs):
        # initialize value functions for multiple tilings and single tiling
        valueFunctions = [TilingsValueFunction(numOfTilings, tileWidth, tilingOffset),
                         ValueFunction(N_STATES // tileWidth)]
        for i in range(0, len(valueFunctions)):
            for episode in range(0, episodes):

                # I use a changing alpha according to the episode instead of a small fixed alpha
                # With a small fixed alpha, I don't think 5000 episodes is enough for so many
                # parameters in multiple tilings.
                # The asymptotic performance for single tiling stays unchanged under a changing alpha,
                # however the asymptotic performance for multiple tilings improves significantly
                alpha = 0.01/50

                # gradient Monte Carlo algorithm
                gradientMonteCarlo(valueFunctions[i], alpha)

                # get state values under current value function
                stateValues = [valueFunctions[i].value(state) for state in states]

                print('run:', run, 'episode:', episode)
                # get the root-mean-squared error
                errors[i][episode] += np.sqrt(np.mean(np.power(trueStateValues[1: -1] - stateValues, 2)))





    # average over independent runs
    errors /= runs

    plt.figure(4)
    for i in range(0, len(labels)):
        plt.plot(errors[i], label=labels[i])
    plt.xlabel('Episodes')
    plt.ylabel('RMSVE')
    plt.legend()




class TilingsValueFunction:
    # @numOfTilings: # of tilings
    # @tileWidth: each tiling has several tiles, this parameter specifies the width of each tile
    # @tilingOffset: specifies how tilings are put together
    def __init__(self, numOfTilings, tileWidth, tilingOffset):
        self.numOfTilings = numOfTilings
        self.tileWidth = tileWidth
        self.tilingOffset = tilingOffset

        # To make sure that each sate is covered by same number of tiles,
        # we need one more tile for each tiling
        self.tilingSize = N_STATES // tileWidth + 1

        # weight for each tile
        self.params = np.zeros((self.numOfTilings, self.tilingSize))

        # For performance, only track the starting position for each tiling
        # As we have one more tile for each tiling, the starting position will be negative
        self.tilings = np.arange(-tileWidth + 1, 0, tilingOffset)

    # get the value of @state
    def value(self, state):
        stateValue = 0.0
        # go through all the tilings
        for tilingIndex in range(0, len(self.tilings)):
            # find the active tile in current tiling
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
            stateValue += self.params[tilingIndex, tileIndex]
        return stateValue

    def update(self, delta, state):

        # each state is covered by same number of tilings
        # so the delta should be divided equally into each tiling (tile)
        delta /= self.numOfTilings

        # go through all the tilings
        for tilingIndex in range(0, len(self.tilings)):
            # find the active tile in current tiling
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
            self.params[tilingIndex, tileIndex] += delta


#figure9_1()
figure9_10()
plt.show()
