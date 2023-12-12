import numpy as np
from custom_env import CustomEnv


def encode_state(state_dict):
    # This function needs to match exactly how you encoded states during training
    return tuple(state_dict["bobbin_positions"])


def test_model(env, q_table, test_episodes, predefined_states=None):
    for episode in range(test_episodes):
        # Initialize state_dict at the beginning of each episode
        if predefined_states:
            env.set_predefined_state(
                predefined_states[episode % len(predefined_states)]
            )
            state_dict = env.state
        else:
            state_dict = env.reset()

        done = False
        total_reward = 0
        step_count = 0

        print(f"\nTesting Episode {episode + 1}")

        while not done:
            state = encode_state(state_dict)
            # Ensure that the state is in the Q-table
            if state in q_table:
                action = np.argmax(q_table[state])
            else:
                # Default action if the state is not in the table
                action = 0

            new_state_dict, reward, done, _ = env.step(action)
            total_reward += reward
            step_count += 1

            print(f"Step {step_count}: Action taken: {action}")
            env.render()
            print(f"Reward for this step: {reward}, Total Reward: {total_reward}")

            state_dict = new_state_dict

        print(
            f"Test Episode {episode + 1} finished after {step_count} steps. Total Reward: {total_reward}\n"
        )


def main():
    # Load the trained Q-table (ensure it's the correct file)
    q_table = np.load("q_table.npy", allow_pickle=True)

    env = CustomEnv()
    test_episodes = 10
    predefined_states = [
        {
            "bobbin_positions": [2, 1, 3, 0, 0, 0, 0, 0, 0],
            "bobbin_priorities": [
                2,
                3,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # Replace with actual priorities
        },
        # ... other predefined states ...
    ]

    # Add more predefined states as needed

    test_model(env, q_table, test_episodes, predefined_states)

    print("Testing completed.\n")


if __name__ == "__main__":
    main()
