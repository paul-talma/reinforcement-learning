# params for QLearners and RL environments
from typing import NamedTuple
from pathlib import Path


class Params(NamedTuple):
    # framework params
    n_episodes: int
    n_runs: int
    savefig_folder: Path
    roll_window: int

    # agent params
    learning_rate: float
    discount_factor: float
    initial_epsilon: float
    epsilon_decay: float
    final_epsilon: float

    # env params
    seed: int
    is_slippery: bool
    proba_frozen: float
    n_actions: int
    n_states: int
    map_size: int
