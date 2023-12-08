import gym
import numpy as np


class CustomEnv(gym.Env):
    step_count = 0

    def __init__(self):
        super().__init__()

        self.step_count = 0
        self.action_space = gym.spaces.Discrete(
            73
        )  # 9*8 moving actions + 1 'do nothing'
        self.observation_space = gym.spaces.Discrete(9)  # 9 slots

        # Priority of each bobbin (randomly assigned for demonstration)
        self.bobbin_priorities = np.random.randint(1, 9, size=8)

        self.state = self.reset()

    def reset(self):
        # Initial state with priorities
        self.step_count = 0  # Reset step_count
        return [
            self.bobbin_priorities[4],
            self.bobbin_priorities[1],
            self.bobbin_priorities[2],
            self.bobbin_priorities[3],
            0,
            0,
            0,
            0,
            0,
        ]

    def step(self, action):
        done = False
        self.step_count += 1  # Increment step_count
        # If the action is 'do nothing', set done to True and return
        if action == 72:
            done = True
            reward = self.calculate_reward()
            return self.state, reward, done, {}

        available_moves = self.find_available_moves()
        if action in available_moves:
            self._move_bobbin(action)
        else:
            print(f"Chosen action {action} is not a valid move.")
            action = 72  # 'Do nothing' action
            done = True

        reward = self.calculate_reward()
        return self.state, reward, done, {}

    def _move_bobbin(self, action):
        if action == 72:  # 'Do nothing' action
            print("Action taken: Do nothing")
            done = True
            return

        # Find the index of the bobbin to move
        bobbin_index = action // 9
        # Find the destination index
        destination_index = action % 9

        # Move the bobbin to the destination
        self.state[destination_index] = self.state[bobbin_index]
        self.state[bobbin_index] = 0

        print(f"Action taken: Move bobbin {bobbin_index} to {destination_index}")
