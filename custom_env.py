import gym
import numpy as np
import random


class CustomEnv(gym.Env):
    def __init__(self):
        """
        Initialize the environment.
        - Define action and observation spaces.
        - Set initial state and bobbin priorities.
        """
        super().__init__()
        self.action_space = gym.spaces.Discrete(
            82
        )  # Define the action space 9*9+1 0-80 arası hareketler + 81 hareketsizlik
        self.observation_space = gym.spaces.Discrete(9)  # Define the observation space
        self.bobbin_priorities = np.random.randint(
            1, 9, size=9
        )  # Randomly assign bobbin priorities
        self.max_bobbin_count = 8
        self.state = self.reset()

    def set_predefined_state(self, predefined_state):
        """
        Set the environment to a predefined state.

        Args:
            predefined_state (list): The state to set the environment to.
        """
        # Assuming predefined_state is a dictionary like {"bobbin_positions": [...], "bobbin_priorities": [...]}
        if (
            isinstance(predefined_state, dict)
            and "bobbin_positions" in predefined_state
            and "bobbin_priorities" in predefined_state
        ):
            self.state = predefined_state
        else:
            raise ValueError("Invalid predefined state format.")

    def reset(self, bobbin_due_dates=None):
        """
        Reset the environment to its initial state.
        """
        max_attempts = 100

        for _ in range(max_attempts):
            # Generate a random bobbin count
            random_bobbin_count = random.randint(1, self.max_bobbin_count)

            # Generate bobbin IDs and priorities
            bobbin_positions = np.zeros(self.max_bobbin_count + 1, dtype=int)
            bobbin_priorities = np.zeros(self.max_bobbin_count + 1, dtype=int)

            # Assign IDs and priorities to the bobbins
            for i in range(random_bobbin_count):
                bobbin_positions[i] = i + 1  # Bobbin ID (assuming it starts from 1)
                if bobbin_due_dates is None:
                    due_date = np.random.randint(1, 10)
                else:
                    due_date = bobbin_due_dates[i]
                bobbin_priorities[i] = self.dueDateConverter(due_date)

            # Shuffle the bobbin positions
            np.random.shuffle(bobbin_positions)

            # Update the priorities to match the shuffled positions
            sorted_priorities = np.zeros_like(bobbin_priorities)
            for i, position in enumerate(bobbin_positions):
                if position > 0:
                    sorted_priorities[i] = bobbin_priorities[position - 1]

            # Update the environment state
            self.state = {
                "bobbin_positions": bobbin_positions,
                "bobbin_priorities": sorted_priorities,
            }

            # Check if the initial positions are feasible
            if self.checkInitialPositionFeasibility(self.state):
                return self.state

        # If no feasible state is found, default to one bobbin in slot 0
        self.state = {
            "bobbin_positions": np.array([1] + [0] * (self.max_bobbin_count - 1)),
            "bobbin_priorities": np.array(
                [self.dueDateConverter(np.random.randint(1, 10))]
                + [0] * (self.max_bobbin_count - 1)
            ),
        }
        return self.state

    def checkInitialPositionFeasibility(self, state):
        """
        Check if the initial state is feasible

        for all existing bobbin in the initial state:
            if the bobbin is not in the first 4 slots,
                and the bobbin doesn't have coils under it (has to have both),

                return false
        else return true
        """

        """
        Check if the initial state is feasible.
        """
        bobbin_positions = state["bobbin_positions"]
        for i in range(4, 9):
            if (
                i < 7
                and bobbin_positions[i] > 0
                and (bobbin_positions[i - 4] == 0 or bobbin_positions[i - 3] == 0)
            ):
                return False
            if (
                i >= 7
                and bobbin_positions[i] > 0
                and (bobbin_positions[i - 3] == 0 or bobbin_positions[i - 2] == 0)
            ):
                return False

        return True  # Funda

    def dueDateConverter(self, dueDate):
        priority = (
            30
            if dueDate == 1
            else 12
            if dueDate == 2
            else 5
            if dueDate == 3
            else 2
            if dueDate == 4
            else 1
        )
        # Implement the due date conversion logic here
        return priority  # Gürkan

    def step(self, action):
        done = False
        reward = 0

        if action == 81:  # 'Do nothing' action
            done = True
        else:
            from_slot, to_slot = divmod(action, 9)

            # Attempt to move the bobbin and check if it was successful
            if not self.move_bobbin(from_slot, to_slot):
                # The move was not valid; you can choose to try another action here
                print(f"Trying another action due to invalid move.")
                # Implement logic for choosing another action if needed

        # Calculate reward and other metrics as before
        reward = self.calculate_reward()
        return self.state, reward, done, {}

    def move_bobbin(self, from_slot, to_slot):
        """
        Move a bobbin from one slot to another.
        - Update the state accordingly.
        """
        # Check if the move is valid
        if not self.is_move_valid(from_slot, to_slot):
            print(f"Invalid move from slot {from_slot + 1} to slot {to_slot + 1}")
            return False  # Indicates the move was not successful

        # If the move is valid, perform it
        # Move the bobbin ID
        self.state["bobbin_positions"][to_slot] = self.state["bobbin_positions"][
            from_slot
        ]
        self.state["bobbin_positions"][from_slot] = 0

        # Move the bobbin priority
        self.state["bobbin_priorities"][to_slot] = self.state["bobbin_priorities"][
            from_slot
        ]
        self.state["bobbin_priorities"][from_slot] = 0

        print(f"Moved bobbin from slot {from_slot + 1} to slot {to_slot + 1}")
        return True  # Indicates the move was successful

    def is_move_valid(self, from_slot, to_slot):
        bobbin_positions = self.state["bobbin_positions"]

        return (
            bobbin_positions[from_slot] > 0
            and self._is_bobbin_free(from_slot)
            and bobbin_positions[to_slot] == 0
            and self._is_slot_available(to_slot, from_slot)
        )

    def _is_bobbin_free(self, slot):
        bobbin_positions = self.state["bobbin_positions"]

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

        return all(bobbin_positions[s] == 0 for s in blocking_slots.get(slot, []))

    def _is_slot_available(self, slot, from_slot=None):
        """
        Check if a slot is available for a bobbin to be moved to.
        """
        bobbin_positions = self.state["bobbin_positions"]

        # Slots 1 to 4 are available if they are empty
        if slot < 4:
            return bobbin_positions[slot] == 0

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
                if bobbin_positions[required_slot] == 0 or (
                    from_slot is not None and required_slot == from_slot
                ):
                    return False
            return True

        return True  # Funda

    def calculate_reward(self):
        reward = 0
        bobbin_positions = self.state["bobbin_positions"]
        bobbin_priorities = self.state["bobbin_priorities"]

        for i in range(len(bobbin_positions)):
            if bobbin_positions[i] > 0:  # Check if there's a bobbin in the slot
                priority = bobbin_priorities[i]
                blocking_slots = self._get_blocking_slots(i)
                blocking_bobbins = len(blocking_slots)
                reward += priority * (11 - 2 * blocking_bobbins)

        return reward

    def _get_blocking_slots(self, slot):
        bobbin_positions = self.state["bobbin_positions"]
        blocking_slots = []

        blocking_slots_mapping = {
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

        if slot in blocking_slots_mapping:
            for blocking_slot in blocking_slots_mapping[slot]:
                if bobbin_positions[blocking_slot] > 0:
                    blocking_slots.append(blocking_slot)

        return blocking_slots

    def render(self, mode="human"):
        bobbin_positions = self.state["bobbin_positions"]
        bobbin_priorities = self.state["bobbin_priorities"]

        # Formatting the state for display
        formatted_state = [
            f"{priority:02d}" if position > 0 else "00"
            for position, priority in zip(bobbin_positions, bobbin_priorities)
        ]

        print(f"Current State: ")
        print("Layer 3:      ", formatted_state[7:9])
        print("Layer 2:   ", formatted_state[4:7])
        print("Layer 1:", formatted_state[0:4])

    def find_available_moves(self):
        available_moves = []
        bobbin_positions = self.state["bobbin_positions"]

        # Define a list of moves to exclude
        excluded_moves = [4, 13, 14, 23, 24, 33, 67, 68, 77, 78]

        for from_slot in range(9):
            for to_slot in range(9):
                action = from_slot * 9 + to_slot
                if (
                    from_slot != to_slot
                    and bobbin_positions[from_slot] > 0
                    and self._is_bobbin_free(from_slot)
                    and bobbin_positions[to_slot] == 0
                    and self._is_slot_available(to_slot, from_slot)
                ):
                    # Exclude specific actions
                    if action not in excluded_moves:
                        available_moves.append(action)

        # Add the 'do nothing' action
        available_moves.append(81)

        return available_moves
