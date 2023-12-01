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

        from_slot = action // 8  # Determine the source slot
        to_slot = action % 8  # Determine the target slot

        # Check if the action is a valid move
        if (
            from_slot < len(self.state)
            and self.state[from_slot] > 0
            and self._is_bobbin_free(from_slot)
        ):
            if (
                to_slot < len(self.state)
                and self.state[to_slot] == 0
                and self._is_slot_available(to_slot, from_slot)
            ):
                # Perform the move
                self.state[to_slot] = self.state[from_slot]
                self.state[from_slot] = 0
                print(f"Moved bobbin from slot {from_slot + 1} to slot {to_slot + 1}")
            else:
                print(f"Target slot {to_slot + 1} is not available for movement.")
        else:
            print(f"No bobbin in slot {from_slot + 1} to move or action out of range.")

    def find_available_moves(self):
        available_moves = []
        for from_slot in range(9):
            for to_slot in range(9):
                if (
                    from_slot != to_slot
                    and self.state[from_slot] > 0
                    and self._is_bobbin_free(from_slot)
                    and self.state[to_slot] == 0
                    and self._is_slot_available(to_slot, from_slot)
                ):
                    available_moves.append(from_slot * 8 + to_slot)
        return available_moves

    def _is_bobbin_free(self, slot):
        # Mapping of slots to the slots that need to be empty for the bobbin to be free
        blocking_requirements = {
            0: [4, 7],
            1: [4, 5, 7, 8],
            2: [5, 6, 7, 8],
            3: [6, 8],
            4: [7],
            5: [7, 8],
            6: [8],
            7: [],
            8: [],
        }

        # Check if all the required slots are empty
        if slot in blocking_requirements:
            required_empty_slots = blocking_requirements[slot]
            # Bobbin is free if all required slots are empty
            return all(self.state[s] == 0 for s in required_empty_slots)

        # If the slot is not in the defined range (for safety)
        return False

    def _is_slot_available(self, slot, action=None):
        # If action is None or invalid, no movement is happening
        if action is None or action < 0 or action > 72:
            return False

        # Determine the source slot based on the action
        from_slot = action

        # If the target slot is the same as the source slot, the move is not valid

        # Mapping of slots to the slots they should land on
        landing_slots_requirements = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [0, 1],
            5: [1, 2],
            6: [2, 3],
            7: [4, 5],
            8: [5, 6],
        }

        # Check if the slot is available based on the bobbins in the slots it is landing on
        if slot in landing_slots_requirements:
            required_slots = landing_slots_requirements[slot]
            # Slot is available if all required slots have bobbins
            for s in required_slots:
                if self.state[s] == 0 and self.state[s + 1] == 0:
                    return False
            return True

        # If the slot is one of the bottom layer slots (0 to 3), it is available if it is empty

        # Check if the slot is available based on the bobbins in the slots it is landing on
        if slot in landing_slots_requirements:
            required_slots = landing_slots_requirements[slot]

            # Slot is available if all required slots have bobbins and are not the source of the current action
            for required_slot in required_slots:
                if self.state[required_slot] == 0 or required_slot == from_slot:
                    return False
            return True

        # If the slot is one of the bottom layer slots (0 to 3), it is available if it is empty
        if slot in range(4):
            return self.state[slot] == 0

        # If the slot is not in the defined range (for safety)
        return False

    def calculate_reward(self):
        reward = 0
        # Example: Iterate through each bobbin and calculate its contribution to the reward
        for i in range(len(self.state)):
            if self.state[i - 1] > 0:  # If there is a bobbin
                blocking_bobbins = self._count_blocking_bobbins(i - 1)
                priority = self.bobbin_priorities[i - 1]
                reward += priority * (11 - 2 * blocking_bobbins)

        reward -= self.step_count + 2  # Add penalty for each step taken
        return reward

    def _count_blocking_bobbins(self, slot):
        blocking_bobbins = 0

        # Mapping of which slots block which other slots
        blocking_slots = {
            0: [4, 7],
            1: [4, 5, 7, 8],
            2: [5, 6, 7, 8],
            3: [6, 8],
            4: [7],
            5: [7, 8],
            6: [8],
            7: [],
            8: [],
        }

        # Check if the slot is in the bottom layer and count blocking bobbins
        if slot in blocking_slots:
            for blocking_slot in blocking_slots[slot]:
                if self.state[blocking_slot] > 0:  # Adjusting index for 0-based array
                    blocking_bobbins += 1
                    # Count bobbins that are blocking the blockers
                    additional_blockers = self._count_blocking_bobbins(blocking_slot)
                    blocking_bobbins += additional_blockers

        return blocking_bobbins

    def render(self, mode="human"):
        print(f"Current State: {self.state}")
        print("")
        print("Layer 1:", self.state[0:4])
        print("Layer 2:", self.state[4:7])
        print("Layer 3:", self.state[7:9])
        print("")


if __name__ == "__main__":
    env = CustomEnv()
    total_episodes = 80

    for episode in range(total_episodes):
        state = env.reset()
        done = False
        step_count = 0
        total_reward = 0

        print(f"\nStarting Episode {episode + 1}")

        while not done:
            # Find available moves
            available_moves = env.find_available_moves()

            # Choose a random action from available moves, or 'do nothing' if none are available
            action = np.random.choice(available_moves) if available_moves else 72

            new_state, reward, done, _ = env.step(action)
            total_reward += reward

            print(f"Step {step_count + 1}: Action taken: {action}")
            env.render()
            print(f"Reward for this step: {reward}, Total Reward: {total_reward}")

            step_count += 1
            if step_count > 100:
                break

        print(
            f"Episode {episode + 1} finished after {step_count} steps. Total Reward: {total_reward}\n"
        )
