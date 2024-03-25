# reinforcement-learning
Implementations of reinforcement learning algorithms, including Deep Q-Learning

toy_text contains warm-up reinforcement learning environments. The small state space allows classic Q-Learning to succeed in solving the environments.

classic_control contains environments with (in-principle) continuous state spaces, requiring the use of deep Q-learning methods to solve.

To run one of the agents, first run the env_main.py file, to train the agent, then run the env_testing.py file to observe the agent's performance. For example, run cartpole_main.py, then cartpole_testing.py. The main.py file saves the trained model in the corresponding saved_models folder.
