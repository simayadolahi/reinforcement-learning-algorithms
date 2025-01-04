from epsilon_greedy import multi_armed_bandit_epsilon_greedy
from optimistic import multi_armed_bandit_optimistic
from non_satationary import multi_armed_bandit_ucb_1, multi_armed_bandit_ucb_2
from ucb import multi_armed_bandit_ucb
from gradient_based import multi_armed_bandit_gradient
from matplotlib import pyplot as plt
import numpy as np

# Define the number of bandit arms, epsilon values, and number of steps
num_bandits = 10  # Number of arms (slot machines)
num_steps = 1000  # Number of steps per simulation
num_cases = 500  # Number of simulations for averaging
c = 1
alpha = 0.1


############################# non-statianary UCB1 ########################

avg_reward = np.zeros(num_steps)
print(f"Running simulations for non-statianary UCB1")

# Perform multiple simulations
for i in range(num_cases):
    reward_history = multi_armed_bandit_ucb_1(num_bandits, num_steps, c)
    
    # Accumulate rewards for averaging
    for j in range(num_steps):
        avg_reward[j] += reward_history[j]
    
    # Compute the average reward across all simulations
    

avg_reward /= num_cases
    
    # Plot the result for this epsilon
plt.plot(avg_reward, label=f"non-statianary UCB1")

############################# non-statianary UCB2 ########################

avg_reward = np.zeros(num_steps)
print(f"Running simulations for non-statianary UCB2")

# Perform multiple simulations
for i in range(num_cases):
    reward_history = multi_armed_bandit_ucb_2(num_bandits, num_steps, c)
    
    # Accumulate rewards for averaging
    for j in range(num_steps):
        avg_reward[j] += reward_history[j]
    
    # Compute the average reward across all simulations
    

avg_reward /= num_cases
    
    # Plot the result for this epsilon
plt.plot(avg_reward, label=f"non-statianary UCB2")

# Add plot labels and legend
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Epsilon-Greedy Performance")
plt.legend()
plt.show()