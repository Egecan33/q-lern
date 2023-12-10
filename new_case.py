import gym
import numpy as np
import random


class CustomEnv(gym.Env):
    """
    Custom Environment for coil stacking, compatible with OpenAI Gym.
    """

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
        self.state = self.reset()

    def reset(self, bobbin_due_dates=None):
        """
        Reset the environment to its initial state.
        - Reset bobbin positions, priorities, and count.
        - Return the initial state for the next episode.
        """
        # Define the initial state here
        max_bobbin_count = 9  # Set the maximum possible bobbin count
        random_bobbin_count = random.randint(
            1, max_bobbin_count
        )  # Generate a random bobbin count

        bobbin_ids = [f"Bobbin_{i}" for i in range(random_bobbin_count)]

        if bobbin_due_dates is None:
            bobbin_due_dates = np.random.randint(1, 10, size=random_bobbin_count)

        bobbin_positions = np.zeros(
            random_bobbin_count, dtype=int
        )  # Reset bobbin positions to zeros initially
        bobbin_dict = dict(zip(bobbin_ids, bobbin_due_dates))

        # Update the environment state with bobbin dictionary, positions, and count
        self.state = {
            "bobbin_positions": bobbin_positions,
            "bobbin_due_dates": bobbin_dict,
            "bobbin_count": random_bobbin_count,
            # convert bobbin due dates to priorities
            # state should be an array of lenght 9 always
            # if there are less than 9 bobbins, the rest of the array should be filled with 0s
            # if there are 9 or 0 bobbins this is not a valid problem so we should not encounter these cases
            # becasue we changed the datatype of bobbin_due_dates to dictionary, we need to convert it back to array to move but we can use this if we change all the neccesary spots in the code
        }

        return self.state  # Pınar

    def checkInitialPositionFeasibility(self, state):
        """
        Check if the initial state is feasible

        for all existing bobbin in the initial state:
            if the bobbin is not in the first 4 slots,
                and the bobbin doesn't have coils under it (has to have both),

                return false
        else return true
        """
        # Implement the feasibility check logic here
        for i in range(4, 9):
            if i < 7 and state[i] > 0 and (state[i - 4] == 0 or state[i - 3] == 0):
                return False
            if i >= 7 and state[i] > 0 and (state[i - 3] == 0 or state[i - 2] == 0):
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
        self.state[to_slot], self.state[from_slot] = (
            self.state[from_slot],
            0,
            
        )  ## explore how this will work with our dicinary format of bobbins

        print(f"Moved bobbin from slot {from_slot + 1} to slot {to_slot + 1}")
        return True  # Indicates the move was successful
        # Implement the bobbin moving logic here #İzel

    def is_move_valid(self, from_slot, to_slot):
        return (
            self.state[from_slot] > 0
            and self._is_bobbin_free(from_slot)
            and self.state[to_slot] == 0
            and self._is_slot_available(to_slot)
        )

    def _is_bobbin_free(self, slot):
        """
        Check if a bobbin is free to be moved.
        - Determine if the bobbin is not blocked by others.
        - Return True if free, False otherwise.
        """
        # Implement the logic to check if a bobbin is free
        return True  # Pınar

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

        return True  # Funda

    def calculate_reward(self):
        """
        Calculate the reward after an action is taken.
        - Define the reward strategy based on the current state.
        - Return the calculated reward.
        """
        # count step puan kırmaca
        # Implement the reward calculation logic here
        return reward  # Pınar + Selin

    def _get_blocking_slots(self, slot):
        """
        Get the slots that block a given slot.
        - Return a list of blocking slots.
        """
        # Define the blocking slots here
        return []  # Pınar +Selin

    def render(self, mode="human"):
        print("\nCurrent State of Environment:")
        print("-" * 30)

        # Layer 3 (Top Layer)
        print("     | ", end="")
        for slot in self.state[7:9]:
            if slot > 0:
                print(f"{slot:02d} | ", end="")
            else:
                print("   | ", end="")
        print()  # New line after top layer

        # Layer 2 (Middle Layer)
        print("  | ", end="")
        for slot in self.state[4:7]:
            if slot > 0:
                print(f"{slot:02d} | ", end="")
            else:
                print("   | ", end="")
        print()  # New line after middle layer

        # Layer 1 (Bottom Layer)
        print("| ", end="")
        for slot in self.state[0:4]:
            if slot > 0:
                print(f"{slot:02d} | ", end="")
            else:
                print("   | ", end="")
        print()  # New line after bottom layer

        print("-" * 30)
        print("Bobbin Priorities:", self.bobbin_priorities)
        print("-" * 30)
        # İzel

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
                    available_moves.append(81)  # do nothing
        return available_moves


# Rest of your Q-learning implementation goes here
# ... [CustomEnv class definition as provided earlier] ...

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

# ... [Previous code including CustomEnv class and Q-learning training] ...


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


# Testing the model
test_episodes = 10  # Number of test episodes
# Define your predefined states/scenarios if any, for testing
predefined_states = [
    # Define states like [state1, state2, ...]
]

test_model(env, q_table, test_episodes, predefined_states)

print("Testing completed.\n")
