import numpy as np

#This class simulates a single "arm" of a bandit machine.
class Bandit():
    """
        Attributes:
        q (float): The true value (mean of the reward distribution) of the bandit.
        Q (float): The estimated value of the bandit, initially set to 0.
        n (int): The number of times the bandit has been pulled (used for updating Q).
    """
    # Initializes a new Bandit instance with a given true value and initial estimate.
    def __init__(self, q:float, Q:float):
        self.q = q
        self.Q = Q
        self.n = 0
    # String representation of the Bandit instance
    def __str__(self):
        return f"q:{self.q :.2f} | Q:{self.Q :.2f} | n:{self.n}"
    # Simulates pulling the bandit arm, returning a reward sampled from a normal distribution.
    def pull(self, std: float = 1):
        return np.random.normal(self.q, std)
    
    # Updates the estimated value of the bandit based on the observed reward.
    def update(self, r):
        self.n += 1
        self.Q = self.Q +(1.0/self.n)*(r - self.Q)



class GradientBandit():
 
    def __init__(self, q:float, H:float):
        self.q = q
        self.H = H
        self.n = 0
    def __str__(self):
        return f"q:{self.q :.2f} | H:{self.H :.2f} | n:{self.n}"
    
    def pull(self, std: float = 1):
        return np.random.normal(self.q, std)
    
    def update(self, new_H):
        self.n += 1
        self.H = new_H


class NSBandit():
    """
        Attributes:
        q (float): The true value (mean of the reward distribution) of the bandit.
        Q (float): The estimated value of the bandit, initially set to 0.
        n (int): The number of times the bandit has been pulled (used for updating Q).
    """
    # Initializes a new Bandit instance with a given true value and initial estimate.
    def __init__(self, q:float, Q:float, alpha):
        self.q = q
        self.Q = Q
        self.n = 0
        self.alpha = alpha
    # String representation of the Bandit instance
    def __str__(self):
        return f"q:{self.q :.2f} | Q:{self.Q :.2f} | n:{self.n}"
    # Simulates pulling the bandit arm, returning a reward sampled from a normal distribution.
    def pull(self, std: float = 1):
        return np.random.normal(self.q, std)
    
    # Updates the estimated value of the bandit based on the observed reward.
    def update(self, r):
        self.n += 1
        self.Q = self.Q + self.alpha*(r - self.Q)

def simple_bandit(q):
    return Bandit(q = q, Q = 0)

def optimistic_bandit(q, Q):
    return Bandit(q = q, Q = Q)

def simple_gradientbandit(q):
    return GradientBandit(q = q, H = 0)

def simple_ns_bandit(q):
    return NSBandit(q, Q=0, alpha=0.1)



