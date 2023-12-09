import gym
import numpy as np


class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(
            73
        )  # 9*8 moving actions + 1 'do nothing'
        self.observation_space = gym.spaces.Discrete(9)  # 9 slots

        # Priority of each bobbin (randomly assigned for demonstration)
        self.bobbin_priorities = np.random.randint(1, 9, size=9)
        self.state = self.reset()

    def reset(self):
        # self.bobbin_priorities = np.random.randint(1, 9, size=9)
        self.state = [
            self.bobbin_priorities[0],
            self.bobbin_priorities[1],
            self.bobbin_priorities[2],
            0,
            self.bobbin_priorities[3],
            self.bobbin_priorities[4],
            0,
            0,
            0,
        ]
        return self.state

    def step(self, action):
        done = False
        if action == 72:  # 'Do nothing' action
            done = True
        else:
            from_slot, to_slot = divmod(action, 9)
            if self.is_move_valid(from_slot, to_slot):
                self.move_bobbin(from_slot, to_slot)
            else:
                print(f"Invalid move from {from_slot} to {to_slot}")

        reward = self.calculate_reward()
        return self.state, reward, done, {}

    def move_bobbin(self, from_slot, to_slot):
        self.state[to_slot], self.state[from_slot] = self.state[from_slot], 0
        print(f"Moved bobbin from slot {from_slot + 1} to slot {to_slot + 1}")

    def is_move_valid(self, from_slot, to_slot):
        return (
            self.state[from_slot] > 0
            and self._is_bobbin_free(from_slot)
            and self.state[to_slot] == 0
            and self._is_slot_available(to_slot)
        )

    def _is_bobbin_free(self, slot):
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
        return all(self.state[s] == 0 for s in blocking_slots.get(slot, []))

    def _is_slot_available(self, slot, from_slot=None):
        # Slots 1 to 4 are available if they are empty
        if slot < 4:
            return self.state[slot] == 0

        # Mapping of slots to the slots they should land on
        landing_slots_requirements = {
            4: [0, 1],
            5: [1, 2],
            6: [2, 3],
            7: [4, 5],
            8: [5, 6],
        }

        # Check if the slot is available based on the bobbins in the slots it is landing on
        if slot in landing_slots_requirements:
            required_slots = landing_slots_requirements[slot]

            # Ensure all required slots have bobbins and our action is not moving one of those
            for required_slot in required_slots:
                if self.state[required_slot] == 0 or required_slot == from_slot:
                    return False
            return True

        # If the slot is not in the defined range (for safety)
        return False

    def calculate_reward(self):
        reward = 0
        for i, priority in enumerate(self.bobbin_priorities):
            if self.state[i] > 0:
                blocking_bobbins = sum(
                    1 for s in self._get_blocking_slots(i) if self.state[s] > 0
                )
                reward += priority * (11 - 2 * blocking_bobbins)

        return reward

    def _get_blocking_slots(self, slot):
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
        return blocking_slots.get(slot, [])

    def render(self, mode="human"):
        print(f"Current State: {self.state}")
        print("Layer 1:", self.state[0:4])
        print("Layer 2:", self.state[4:7])
        print("Layer 3:", self.state[7:9])

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
                    action = from_slot * 9 + to_slot
                    available_moves.append(action)
        return available_moves


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
                print("Maximum steps reached for this episode.")
                break

        print(
            f"Episode {episode + 1} finished after {step_count} steps. Total Reward: {total_reward}\n"
        )
