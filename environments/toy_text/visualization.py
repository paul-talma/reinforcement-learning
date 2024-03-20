# visualization tools for RL agents
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def postprocess(episodes, params, rewards, episode_lengths, map_size):
    """Convert learning statistics into dataframe"""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(),
            "Steps": episode_lengths.flatten(),
        }
    )
    res["Cumulative rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(
        data={"Episodes": episodes, "Steps": episode_lengths.mean(axis=1)}
    )
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])

    return res, st


# def make_greedy_policy(q_table, map_size):
#     greedy_pol = np.empty(q_table.flatten().shape)
#     for idx, val in enumerate(greedy_pol.flatten()):
#         if max(q_table.flatten()[idx]) > 0:
#             greedy_pol[idx] =
#     return np.argmax(q_table, axis=1).reshape(map_size, map_size)


def q_table_direction_map(q_table, map_size):
    """Converts Q-values to greedy policy, represents greedy action as arrow"""
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    max_vals = q_table.max(axis=1).reshape(map_size, map_size)
    best_actions = np.argmax(q_table, axis=1).reshape(map_size, map_size)
    directions = np.empty(best_actions.flatten().shape, dtype=str)
    for idx, val in enumerate(best_actions.flatten()):
        if max_vals.flatten()[idx] > 0:
            directions[idx] = directions[val]
    directions = directions.reshape(map_size, map_size)
    return max_vals, directions


# display final frame on left, heatmap and policy on right
def plot_q_val_map(q_table, env, map_size, params):
    max_vals, directions = q_table_direction_map(q_table, map_size)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # display final frame
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # display policy
    sns.heatmap(
        max_vals,
        annot=directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
    ).set(title="Learned Q-values\nArrows represent best action")
    img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


# display distribution of actions and states
def plot_states_actions_distribution(states, actions, map_size, params):
    """Plot the distributions of states and actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()
