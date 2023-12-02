Coil Stacking Optimization Using Q-Learning

This repository contains the implementation of a Q-Learning algorithm applied to a coil stacking optimization problem. The project aims to develop an efficient strategy for arranging coils in a three-dimensional stacking space, prioritizing high-urgency items while minimizing movements.

Overview

The project utilizes Q-Learning, a model-free reinforcement learning algorithm, to determine the optimal action-selection policy for organizing coils based on their priority levels. This method is particularly adept at handling complex and dynamic environments like coil stacking, where the state and action spaces are large and not well-defined.

Repository Contents

clean_case.py: The primary script containing the latest implementation of the custom environment for coil stacking, integrated with the Q-Learning algorithm. This is the main focus of the repository and is used for training the model.
simple_case.py: An earlier version of the environment that served as a foundational step towards the current implementation. While it's not the focus of current development, it provides valuable insights into the evolution of the project.
qlearn.py: The main script where the Q-Learning algorithm is implemented and interfaced with the environment defined in clean_case.py.
Environment Setup

The environment is structured as a series of layered grids, each representing a level within the stacking space. The stability of higher-level coils depends on the supporting coils beneath them. The custom environment class (CustomEnv) handles the dynamics of coil movement, validity of actions, reward calculation, and state transitions.

Q-Learning Implementation

The Q-Learning algorithm is implemented to estimate the value of each action within a specific state. It updates Q-values based on the rewards received from the environment and the predicted future rewards, balancing the exploration of new actions with the exploitation of known actions.

Running the Project

Ensure you have Python and necessary libraries installed (e.g., gym, numpy).
Clone the repository.
Run qlearn.py to start the training process using the clean_case environment.
You can also explore the simple_case.py for understanding the initial model structure.