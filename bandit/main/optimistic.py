import numpy as np
from bandit import optimistic_bandit

def multi_armed_bandit_optimistic(num_bandits, epsilon, num_steps, Q_init):
    bandits = [optimistic_bandit(q=np.random.normal(0, 1), Q = Q_init) for i in range(num_bandits)]
    reward_history = []

    for i in range(num_steps):
        if np.random.uniform(0, 1) < epsilon:
            #exploration
            a = np.random.randint(0, num_bandits)
        else:
            #exploitation
            a = np.argmax([b.Q for b in bandits])
        r = bandits[a].pull() # Pulls the selected arm (a) and receives a reward (r) based on its true value (q).
        bandits[a].update(r)
        reward_history.append(r)

    return reward_history

