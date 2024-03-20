import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

from gymnasium.wrappers import RecordEpisodeStatistics

from q_agent_template import QAgent
from quickplot import quickplot

from IPython.display import display

train_env = gym.make("FrozenLake-v1", is_slippery=False)
test_env = gym.make("FrozenLake-v1", is_slippery=False)

# hyperparameters
n_episodes = 5000
start_epsilon = 1.0
final_epsilon = 0.01
epsilon_decay = 0.0015
learning_rate = 0.8
discount_factor = 0.99


agent = QAgent(
    train_env,
    learning_rate,
    start_epsilon,
    epsilon_decay,
    final_epsilon,
    discount_factor,
)

# train agent
train_env = RecordEpisodeStatistics(train_env, deque_size=n_episodes)

epsilon_values = []  # tracks epsilon decay
episode_lengths = []  # tracks episode length


for e in trange(n_episodes):
    state, _ = train_env.reset()
    done = False
    episode_length = 0

    while not done:
        episode_length += 1

        action = agent.epsilon_greedy(state)
        next_state, reward, terminated, truncated, _ = train_env.step(action)
        agent.learn(state, action, reward, next_state, terminated)

        done = terminated or truncated
        state = next_state

    epsilon_values.append(agent.epsilon)
    episode_lengths.append(episode_length)
    agent.decay_epsilon()

train_env.close()


# plotting
def moving_average(data, len_window):
    data_sum = np.cumsum(data)
    return (data_sum[len_window:] - data_sum[:-len_window]) / len_window


returns = train_env.return_queue
returns = moving_average(returns, 100)

fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

axs[0].plot(epsilon_values)
axs[0].set_xlabel("Episodes")
axs[0].set_ylabel("Epsilon")

axs[1].plot(returns)
axs[1].set_xlabel("Episodes")
axs[1].set_ylabel("Returns")

axs[2].plot(episode_lengths)
axs[2].set_xlabel("Episodes")
axs[2].set_ylabel("Lengths")

plt.tight_layout()
plt.show()


# testing parameters
n_tests = 10
successes = 0

# evaluate agent
for test in trange(n_tests):
    state, _ = test_env.reset()
    done = False

    while not done:
        action = agent.greedy(state)
        next_state, reward, terminated, truncated, _ = test_env.step(action)

        successes += reward
        state = next_state
        done = terminated or truncated

test_env.close()

print(f"The agent has passed {successes/n_tests * 100}% of the tests!")

# display greedy strategy
view_env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
state, _ = view_env.reset()
done = False

while not done:
    action = agent.greedy(state)
    next_state, reward, terminated, truncated, _ = view_env.step(action)

    state = next_state
    done = terminated or truncated

view_env.close()
