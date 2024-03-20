# params for QLearners and RL environments
from typing import NamedTuple
from pathlib import Path


class Params(NamedTuple):
    # framework params
    n_episodes: int
    n_runs: int
    roll_window: int

    # agent params
    learning_rate: float
    discount_factor: float
    initial_epsilon: float
    epsilon_decay: float
    final_epsilon: float

    # env params
    seed: int
    n_actions: int
    n_states: int


class FrozenLakeParams(Params):
    map_size: int
    is_slippery: bool
    proba_frozen: float
    savefig_folder: Path


class CartPoleParams(Params):
    state_space_shape: tuple
    buffer_capacity: int
    batch_size: int
