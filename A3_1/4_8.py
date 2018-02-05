import time, sys
import numpy as np

head_probability = 0.25 # head_probability: floating point
num_total_states = 100 # num_total_states: integer
discount_rate = 1
theta = 0.0000001

V = np.zeros(num_total_states+ 1)
final_policy = np.zeros(num_total_states - 1)
final_value = np.zeros((num_total_states - 1, 4))
current_action = np.zeros(1)

max_value_action = 0
sweep = 0

if __name__ == '__main__':

    while True:

        delta = 0

        for s in range(1,num_total_states):
            v = V[s]
            actions_list = []
            for action in range(1, min(s, 100-s)+1):
                if s + action >= 100:
                    current_action[0] = head_probability*(1+discount_rate * V[action+s]) + (1 - head_probability) * (0 + discount_rate * V[s-action])
                else:
                    current_action[0] = head_probability*(0+discount_rate * V[action+s]) + (1 - head_probability) * (0 + discount_rate * V[s-action])

                actions_list.append(current_action[0])
            V[s] = max(actions_list)
            final_policy[s-1] = actions_list.index(V[s])+1
            delta = max(delta, abs(v - V[s]))

            if sweep < 3:
                final_value[s-1][sweep] = V[s]
            else:
                final_value[s-1][3] = V[s]


        if delta < theta:
            break
        sweep +=1

    np.save("Value_function",s)
    np.save("final_policy",final_policy)



