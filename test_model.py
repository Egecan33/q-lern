import numpy as np
from custom_env import CustomEnv


q_table = np.load("q_table.npy")  # Load the Q-table


def test_specific_state(env, q_table, predefined_state):
    # Set the environment to the predefined state
    env.set_predefined_state(predefined_state)
    state_dict = env.set_predefined_state(predefined_state)

    done = False
    total_reward = 0
    step_count = 0

    while not done:
        state = encode_state(state_dict)
        # Choose the best action based on the Q-table
        if state in q_table:
            q_values = q_table[state]
            action = np.argmax(q_values)
        else:
            # Default action if the state is not in the Q-table
            action = 81

        # Take the action and observe the outcome
        new_state_dict, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1

        # Print the step, action, and current state
        print(f"Step {step_count}: Action taken: {action}")
        env.render()
        print(f"Reward for this step: {reward}, Total Reward: {total_reward}")

        # Update the state
        state_dict = new_state_dict

    print(f"Test finished after {step_count} steps. Total Reward: {total_reward}")


def main():
    q_table = np.load("q_table.npy", allow_pickle=True)
    env = CustomEnv()

    predefined_state = {
        "bobbin_positions": [1, 1, 0, 0, 1, 0, 0, 0, 0],
        "bobbin_priorities": [2, 2, 0, 0, 10, 0, 0, 0, 0],
    }

    test_specific_state(env, q_table, predefined_state)

    print("Testing completed.\n")


if __name__ == "__main__":
    main()
