# params for QLearners and RL environments
from typing import NamedTuple
from pathlib import Path


class Params(NamedTuple, env):
    # framework params
    n_episodes: int
    n_runs: int
    savefig_folder: Path

    # agent params
    learning_rate: float
    discount_factor: float
    initial_epsilon: float
    epsilon_decay: float
    final_epsilon: float

    # env params
    seed: int
    n_actions = env.action_space.n
    n_states = env.observation_space.n
