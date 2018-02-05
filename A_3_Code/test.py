import numpy as np
from utils import rand_in_range, rand_un
import time

actions_dic = {0:[0, 1], 1:[-1, 1], 2:[-1, 0], 3:[-1,-1], 4:[0,-1], 5:[1,-1], 6:[1,0], 7:[1,1]}

while True:
    action = actions_dic[rand_in_range(7)]

    print(action[0])

    time.sleep(0.5)