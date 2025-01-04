import numpy as np
from bandit import simple_gradientbandit

def softmax(values):
    exp = np.exp(values)
    sum_exp = np.sum(exp)
    pi = exp/sum_exp
    return pi

def multi_armed_bandit_gradient(num_bandits, num_steps, alpha):
    bandits = [simple_gradientbandit(q=np.random.normal(0, 1)) for i in range(num_bandits)]
    reward_history = []
    mean_R = 0

    for i in range(num_steps):
        H_vals = [b.H for b in bandits]
        pi = softmax(H_vals)
        a = np.random.choice(num_bandits, p = pi)

        r = bandits[a].pull()
        reward_history.append(r)

        #update H
        for i, b in enumerate(bandits):
            if i == a:
                new_H = b.H + alpha*(r - mean_R) * (1-pi[i])
            else:
                new_H = b.H - alpha*(r - mean_R)*(pi[i])
            b.update(new_H)

        reward_history.append(r)
        mean_R = np.mean(reward_history)

    return reward_history