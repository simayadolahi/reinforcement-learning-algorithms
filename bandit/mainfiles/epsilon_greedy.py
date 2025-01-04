import numpy as np
from bandit import simple_bandit

def multi_armed_bandit_epsilon_greedy(num_bandits, epsilon, num_steps):
    bandits = [simple_bandit(q=np.random.normal(0, 1)) for i in range(num_bandits)]
    reward_history = []

    for i in range(num_steps):
        if np.random.uniform(0, 1) < epsilon:
            #exploration
            a = np.random.randint(0, num_bandits-1)
        else:
            #exploitation
            a = np.argmax([b.Q for b in bandits])
        r = bandits[a].pull() # Pulls the selected arm (a) and receives a reward (r) based on its true value (q).
        bandits[a].update(r)
        reward_history.append(r)

    return reward_history




