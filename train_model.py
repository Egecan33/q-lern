import numpy as np
import random
from custom_env import CustomEnv  # Import CustomEnv class

# Initialize the environment

# Q-Learning Algorithm Implementation

# Initialize the environment
env = CustomEnv()

# Define Q-learning parameters
learning_rate = 0.1  # Learning rate for Q-learning updates
discount_factor = 0.9  # Discount factor for future rewards
epsilon = 1.0  # Initial exploration rate
max_epsilon = 1.0  # Maximum exploration rate
min_epsilon = 0.01  # Minimum exploration rate
epsilon_decay = 0.0001  # Decay rate for exploration probability
total_episodes = 350  # Total number of episodes for training

# Initialize the Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))  # Egecan

# Q-learning algorithm
for episode in range(total_episodes):
    state = env.reset()  # Reset the environment for a new episode
    done = False  # Initialize 'done' flag to False
    total_reward = 0  # Initialize total reward for the episode

    while not done:
        # Exploration-exploitation trade-off
        if random.uniform(0, 1) < epsilon:
            # Explore: choose a random action
            action = random.choice(
                env.find_available_moves()
            )  # Random action from available moves
        else:
            # Exploit: choose the best feasible action based on Q-table
            q_values = q_table[state]
            sorted_actions = np.argsort(q_values)[
                ::-1
            ]  # Sort actions in descending order of Q-value
            action = 81  # Default action if no feasible action is found

            for potential_action in sorted_actions:
                if potential_action in env.find_available_moves():
                    action = potential_action
                    break

        # Take the action and observe the outcome
        new_state, reward, done, _ = env.step(action)

        total_reward += reward  # değişebilir#

        # Q-learning update rule
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward
            + discount_factor * np.max(q_table[new_state, :])
            - q_table[state, action]
        )

        # Transition to the new state
        state = new_state

    # Reduce epsilon (exploration rate)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
        -epsilon_decay * episode
    )

    # Print episode summary
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

print("Training completed.\n")
