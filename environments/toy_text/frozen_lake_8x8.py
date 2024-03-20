import gymnasium as gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange


import visualization

from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from q_agent_template import QAgent
import params

from IPython.display import display

from pathlib import Path


params = params.FrozenLakeParams(
    # framework params
    n_episodes=2000,
    n_runs=2,
    roll_window=100,
    # agent params
    learning_rate=0.8,
    discount_factor=0.95,
    initial_epsilon=1,
    epsilon_decay=0.99,
    final_epsilon=0.1,
    # env params
    seed=100,
    n_actions=None,
    n_states=None,
    # env subclass params
    map_size=None,
    is_slippery=False,
    proba_frozen=0.9,
    savefig_folder=Path("../_static/img/frozen_lake"),
)

params.savefig_folder.mkdir(parents=True, exist_ok=True)


# running training
def run_training(env):
    returns = np.zeros((params.n_episodes, params.n_runs))
    episode_lengths = np.zeros((params.n_episodes, params.n_runs))
    q_tables = np.zeros((params.n_runs, params.n_states, params.n_actions))
    all_states = []
    all_actions = []
    episodes = np.arange(params.n_episodes)

    for run in trange(params.n_runs):
        agent.reset_q_values()

        for episode in trange(params.n_episodes):
            state, _ = env.reset()
            done = False
            episode_length = 0
            episode_returns = 0

            while not done:
                episode_length += 1

                action = agent.epsilon_greedy(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.learn(state, action, reward, next_state, terminated)

                # record state and action
                all_states.append(state)
                all_actions.append(action)

                episode_returns += reward

                done = terminated or truncated
                state = next_state

            # record episode returns and length
            returns[episode][run] = episode_returns
            episode_lengths[episode][run] = episode_length

            agent.decay_epsilon()
        q_tables[run, :, :] = agent.q_table

    env.close()
    return (
        returns,
        episode_lengths,
        episodes,
        q_tables,
        all_states,
        all_actions,
    )


map_sizes = [6]
res_all = pd.DataFrame()
st_all = pd.DataFrame()

for map_size in map_sizes:
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=params.is_slippery,
        render_mode="rgb_array",
        desc=generate_random_map(
            size=map_size, p=params.proba_frozen, seed=params.seed
        ),
    )

    params = params._replace(n_actions=env.action_space.n)
    params = params._replace(n_states=env.observation_space.n)
    env.action_space.seed(params.seed)

    agent = QAgent(
        env,
        params.learning_rate,
        params.initial_epsilon,
        params.epsilon_decay,
        params.final_epsilon,
        params.n_states,
        params.n_actions,
        params.discount_factor,
    )

    print(f"Map size: {map_size}x{map_size}")
    returns, episode_lengths, episodes, q_tables, all_states, all_actions = (
        run_training(env)
    )

    res, st = visualization.postprocess(
        episodes, params, returns, episode_lengths, map_size
    )
    res_all = pd.concat([res_all, res])
    st_all = pd.concat([st_all, st])
    q_table = q_tables.mean(axis=0)

    visualization.plot_states_actions_distribution(
        states=all_states, actions=all_actions, map_size=map_size, params=params
    )
    visualization.plot_q_val_map(q_table, env, map_size, params=params)

    env.close()
