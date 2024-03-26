# reinforcement-learning
Implementations of reinforcement learning algorithms, including Deep Q-Learning

[toy_text](toy_text/) contains warm-up reinforcement learning environments. The small state space allows classic Q-Learning algorithms to succeed in solving the environments.

[classic_control](classic_control/) contains environments with (in-principle) continuous state spaces and/or action spaces, requiring the use of more sophisticated methods to solve.

  - [cartpole](classic_control/cartpole) features a continuous state space and a discrete (indeed, two-valued) action space. To handle the continuous state space, we use a Deep Q-Network ([Mnih et al. (2016)](https://doi.org/10.1038/nature14236)) featuring an experience replay memory. To improve training stability, we use both a policy network and a target network. Instead of copying the policy network to the target network after a constant number of steps, we have the target network track the policy network by a factor of $\tau = 0.005$ ([Lillicrap et al. (2016)](https://doi.org/10.48550/arXiv.1509.02971)).
  - [acrobot](classic_control/acrobot) is similar to [cartpole](classic_control/cartpole) and the same DQN suffices to solve the environment (albeit requiring about half as many training episodes).
  - [mountain_car_continuous](classic_control/mountain_car_continuous) is significantly more challenging than both [cartpole](classic_control/cartpole) and [acrobot](classic_control/acrobot), as it features both a continuous state space and a continuous action space. To solve this environment, the use of actor-critic methods is required in addition to the aforementioned sophistications ([Lillicrap et al. (2016)](https://doi.org/10.48550/arXiv.1509.02971), [Silver et al. (2014)(http://proceedings.mlr.press/v32/silver14.pdf)]).
    - [mountain_car](classic_control/mountain_car) is like [mountain_car_continuous](classic_control/mountain_car_continuous), except that it has a discrete action space. The continuous state space can be reasonably approximated using discretization methods, allowing us to use traditional Q-learning methods.

To run one of the agents, first run the `env_main.py` file, to train the agent, then run the `env_testing.py` file to observe the agent's performance. For example, run `cartpole_main.py`, then `cartpole_testing.py`. The `env_main.py` file saves the trained model in the corresponding `saved_models` directory.
