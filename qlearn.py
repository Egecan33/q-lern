import gym
import numpy as np
import random
from clean_case import CustomEnv  # Replace with your actual environment class

# Initialize the environment
env = CustomEnv()

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.00001
total_episodes = 3

# Initialize the Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Q-learning algorithm
for episode in range(total_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    print(f"\nStarting Episode {episode + 1}")

    while not done:
        available_moves = env.find_available_moves()
        if random.uniform(0, 1) < epsilon:
            # Explore: choose a random action from the available moves
            action = random.choice(available_moves) if available_moves else 72
        else:
            # Exploit: choose the best action from Q-table
            if available_moves:
                # Filter Q-table for available actions and select the best action
                filtered_q_values = [q_table[state, move] for move in available_moves]
                best_action_index = np.argmax(filtered_q_values)
                action = (
                    available_moves[best_action_index]
                    if best_action_index < len(available_moves)
                    else 72
                )
            else:
                action = 72  # 'Do nothing' action

        # Take action and observe the outcome
        new_state, reward, done, _ = env.step(action)
        total_reward += reward

        # Update Q-table
        if new_state is not None:
            q_table[state, action] = q_table[state, action] + learning_rate * (
                reward
                + discount_factor * np.max(q_table[new_state, :])
                - q_table[state, action]
            )

        # Print step details
        print(f"Step {step_count + 1}: Action taken: {action}")
        env.render()
        print(f"Reward for this step: {reward}, Total Reward: {total_reward}")

        state = new_state
        step_count += 1

        if step_count > 30:
            print("Maximum steps reached for this episode.")
            break

    # Reduce epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
        -epsilon_decay * episode
    )
    print(
        f"Episode {episode + 1} finished after {step_count} steps. Total Reward: {total_reward}\n"
    )

print("Training finished.\n")
