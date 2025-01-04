from epsilon_greedy import multi_armed_bandit_epsilon_greedy
from matplotlib import pyplot as plt
import numpy as np

# Define the number of bandit arms, epsilon values, and number of steps
num_bandits = 10  # Number of arms (slot machines)
epsilons = [0, 0.1, 0.9]  # Different epsilon values for exploration-exploitation tradeoff
num_steps = 1000  # Number of steps per simulation
num_cases = 500  # Number of simulations for averaging



"""
    During each simulation, the agent records a reward for every step (from step 0 to step 999).
    This means for each step index (e.g., step 0, step 1, ..., step 999), there will be 500 reward values—one from each simulation.
"""

# Iterate over epsilon values
for epsilon in epsilons:
    print(f"Running simulations for epsilon = {epsilon}")
    
    # Initialize average reward array
    avg_reward = np.zeros(num_steps)
    
    # Perform multiple simulations
    for i in range(num_cases):
        reward_history = multi_armed_bandit_epsilon_greedy(num_bandits, epsilon, num_steps)
        
        # Accumulate rewards for averaging
        for j in range(num_steps):
            avg_reward[j] += reward_history[j]
    
    # Compute the average reward across all simulations
    avg_reward /= num_cases
    
    # Plot the result for this epsilon
    plt.plot(avg_reward, label=f"Epsilon: {epsilon}")

# Add plot labels and legend
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Epsilon-Greedy Performance")
plt.legend()
plt.show()