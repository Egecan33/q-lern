import numpy as np
from custom_env import CustomEnv  # Import CustomEnv class


def test_model(env, q_table, test_episodes, predefined_states=None):
    """
    Test the trained model with specific scenarios.

    Args:
        env (gym.Env): The custom environment.
        q_table (np.array): Trained Q-table.
        test_episodes (int): Number of episodes to run for testing.
        predefined_states (list): Optional list of predefined states to test.
    """
    for episode in range(test_episodes):
        if predefined_states:
            # Set the environment to a predefined state if provided
            state = predefined_states[episode % len(predefined_states)]
            env.set_state(state)
        else:
            # Reset environment to a random initial state
            state = env.reset()

        done = False
        total_reward = 0
        step_count = 0

        print(f"\nTesting Episode {episode + 1}")

        while not done:
            # Choose action based on the trained Q-table (exploitation only)
            action = np.argmax(q_table[state])

            # Take the action and observe the outcome
            new_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Print step details
            print(f"Step {step_count + 1}: Action taken: {action}")
            env.render()
            print(f"Reward for this step: {reward}, Total Reward: {total_reward}")

            state = new_state
            step_count += 1

        print(
            f"Test Episode {episode + 1} finished after {step_count} steps. Total Reward: {total_reward}\n"
        )


def main():
    # Load the trained Q-table
    q_table = np.load("clean_case.cpython-310.pyc")

    # Initialize the environment
    env = CustomEnv()

    # Define test parameters
    test_episodes = 10
    predefined_states = [
        # Define states like [state1, state2, ...]
    ]

    # Run the testing
    test_model(env, q_table, test_episodes, predefined_states)

    print("Testing completed.\n")


if __name__ == "__main__":
    main()
