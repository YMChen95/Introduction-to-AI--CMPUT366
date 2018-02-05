import numpy as np

V = np.zeros(101)
policy = np.zeros(99)
ph = 0.55
theta = 10**-100
action_value = np.zeros(1)
def Iteration():
    global V, theta
    loop = True
    while loop == True:
        delta = 0
        for s in range(1, 100):
            list1 = []
            v = V[s]
            for action in range(1, min(s, 100-s)+1):
                if s + action >= 100:
                    action_value[0] = ph * (1 + 1 * V[action + s]) + (1 - ph) * (0 + 1 * V[s - action])
                else:
                    action_value[0] = ph * (0 + 1 * V[action + s]) + (1 - ph) * (0 + 1 * V[s - action])

                list1.append(action_value[0])
            V[s] = max(list1)
            policy[s-1] = list1.index(max(list1))+1
            delta = max(delta, abs(v-V[s]))
        if delta < theta:
            loop = False


def save_results(data, data_size, filename): # data: floating point, data_size: integer, filename: string
    with open(filename, "w") as data_file:
        for i in range(data_size):
            data_file.write("{0}\n".format(data[i]))

if __name__ == '__main__':
    Iteration()
    print policy
    save_results(policy, 99, "test.dat")
    save_results(V, 100, "test1.dat")
