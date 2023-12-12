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
epsilon_decay = 0.01  # Decay rate for exploration probability
total_episodes = 500000  # Total number of episodes for training


# Function to encode the state into a unique index
def encode_state(state_dict):
    return tuple(state_dict["bobbin_positions"])


# Initialize the Q-table with a size based on the possible states and actions
# The size and method of initialization might need to be adjusted based on the actual state encoding
q_table = {}

# Q-learning algorithm
for episode in range(total_episodes):
    state_dict = env.reset()  # Reset the environment for a new episode
    state = encode_state(state_dict)
    done = False
    total_reward = 0
    step_count = 0

    print(f"\nStarting Episode {episode + 1}")

    while not done:
        step_count += 1

        # Choose action based on Q-learning
        if random.uniform(0, 1) < epsilon:
            # Exploration: Choose a random action from available moves
            action = random.choice(env.find_available_moves())
        else:
            # Exploitation: Choose the best action based on Q-table
            available_actions = env.find_available_moves()
            q_values = q_table.get(state, np.zeros(env.action_space.n))
            # Sort actions by Q-value but only consider available actions
            sorted_actions = [
                a for a in np.argsort(q_values)[::-1] if a in available_actions
            ]
            action = (
                sorted_actions[0] if sorted_actions else 81
            )  # Default to 'do nothing' if no valid actions

        # Take the action and observe the outcome
        new_state_dict, reward, done, _ = env.step(action)
        new_state = encode_state(new_state_dict)
        total_reward += reward

        # Print the step, action, and current state
        print(f"Step {step_count}: Action taken: {action}")
        env.render()
        print(f"Reward for this step: {reward}, Total Reward: {total_reward}")

        # Update the Q-table
        q_table.setdefault(state, np.zeros(env.action_space.n))
        q_table[state][action] = q_table[state][action] + learning_rate * (
            reward
            + discount_factor
            * np.max(q_table.get(new_state, np.zeros(env.action_space.n)))
            - q_table[state][action]
        )

        # Transition to the new state
        state = new_state

        # Check for maximum step count
        if step_count > 100:
            print("Maximum steps reached for this episode.")
            break

    print(
        f"Episode {episode + 1} finished after {step_count} steps. Total Reward: {total_reward}"
    )

    # Reduce epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
        -epsilon_decay * episode
    )

print("Training completed.\n")
np.save("q_table.npy", q_table)
