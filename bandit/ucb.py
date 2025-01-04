import numpy as np
import math
from bandit import simple_bandit

def multi_armed_bandit_ucb(num_bandits, num_steps, c):
    bandits = [simple_bandit(q=np.random.normal(0, 1)) for i in range(num_bandits)]
    reward_history = []

    for i in range(num_steps):
        ucb_values = [b.Q + c*math.sqrt(math.log(i+1)/(b.n + 1.e-6))  for b in bandits]
        a = np.argmax(ucb_values)
        r = bandits[a].pull() # Pulls the selected arm (a) and receives a reward (r) based on its true value (q).
        bandits[a].update(r)
        reward_history.append(r)

    return reward_history

