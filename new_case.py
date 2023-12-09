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
        self.action_space = gym.spaces.Discrete(73)  # Define the action space
        self.observation_space = gym.spaces.Discrete(9)  # Define the observation space
        self.bobbin_priorities = np.random.randint(
            1, 9, size=9
        )  # Randomly assign bobbin priorities
        self.state = self.reset()

    def reset(self):
        """
        Reset the environment to its initial state.
        - Reset bobbin positions and priorities.
        - Return the initial state for the next episode.
        """
        # Define the initial state here
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
        return True  # Funda

    def dueDateConverter(self, dueDate):
        """
        Due dates→Priority
            1→30
            2→12
            3→5
            4→2
            >=5→1
        """
        # Implement the due date conversion logic here
        return priority  # Gürkan

    def step(self, action):
        """
        Execute one time step within the environment.
        - Apply the given action.
        - Update the environment state.
        - Calculate reward.
        - Check if the episode is done.
        - Return the new state, reward, done, and info.
        """
        # Implement the action logic here
        return self.state, reward, done, {}  # Selin

    def move_bobbin(self, from_slot, to_slot):
        """
        Move a bobbin from one slot to another.
        - Update the state accordingly.
        """
        self.state[to_slot], self.state[from_slot] = self.state[from_slot], 0
        print(f"Moved bobbin from slot {from_slot + 1} to slot {to_slot + 1}")
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
        return True

    def _is_slot_available(self, slot, from_slot=None):
        """
        Check if a slot is available for moving a bobbin into.
        - Verify based on the current state and game rules.
        - Return True if available, False otherwise.
        """
        # Implement the slot availability logic here
        return True

    def calculate_reward(self):
        """
        Calculate the reward after an action is taken.
        - Define the reward strategy based on the current state.
        - Return the calculated reward.
        """
        # Implement the reward calculation logic here
        return reward

    def _get_blocking_slots(self, slot):
        """
        Get the slots that block a given slot.
        - Return a list of blocking slots.
        """
        # Define the blocking slots here
        return []

    def render(self, mode="human"):
        """
        Render the current state of the environment.
        - Display the state in a human-readable format.
        """
        # Implement the rendering logic here

    def find_available_moves(self):
        """
        Find all available moves in the current state.
        - Return a list of possible actions.
        """
        # Implement the logic to find available moves
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
epsilon_decay = 0.00001  # Decay rate for exploration probability
total_episodes = 3500  # Total number of episodes for training

# Initialize the Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

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
            )  # may be a greedy heuristic instead of random
        else:
            # Exploit: choose the best action based on Q-table
            action = np.argmax(q_table[state])

        # Take the action and observe the outcome
        new_state, reward, done, _ = env.step(action)
        total_reward += reward

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
