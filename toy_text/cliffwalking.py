import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from gymnasium.wrappers import RecordEpisodeStatistics

from collections import defaultdict
from tqdm import trange

# from plotting import plot_results


env = gym.make("CliffWalking-v0")


class QLearner:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.gamma = discount_factor

        self.q_vals = defaultdict(lambda: [0] * env.action_space.n)

        self.training_error = []

    def epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.q_vals[state])

    def greedy(self, state):
        return np.argmax(self.q_vals[state])

    def learn(self, state, action, reward, terminated, next_state):
        next_q_val = (not terminated) * max(self.q_vals[next_state])
        temporal_difference = (
            reward + self.gamma * next_q_val - self.q_vals[state][action]
        )

        self.q_vals[state][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class SARSALearner:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.gamma = discount_factor

        self.q_vals = defaultdict(lambda: [0] * env.action_space.n)

        self.training_error = []

    def epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.q_vals[state])

    def greedy(self, state):
        return np.argmax(self.q_vals[state])

    def learn(self, state, action, reward, terminated, next_state):
        next_action = self.epsilon_greedy(next_state)
        next_q_val = (not terminated) * self.q_vals[next_state][next_action]
        temporal_difference = (
            reward + self.gamma * next_q_val - self.q_vals[state][action]
        )

        self.q_vals[state][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# hyperparameters
learning_rate = 0.1
n_episodes = 1_000
start_epsilon = 1
epsilon_decay = 0.05
final_epsilon = 0.1
discount_factor = 0.95

Qagent = QLearner(
    learning_rate,
    start_epsilon,
    epsilon_decay,
    final_epsilon,
    discount_factor,
)

Sagent = SARSALearner(
    learning_rate,
    start_epsilon,
    epsilon_decay,
    final_epsilon,
    discount_factor,
)

# train Q
envQ = RecordEpisodeStatistics(env, deque_size=n_episodes)

for episode in trange(n_episodes):
    state, info = envQ.reset()
    done = False

    while not done:
        action = Qagent.epsilon_greedy(state)
        next_state, reward, terminated, truncated, info = envQ.step(action)
        Qagent.learn(state, action, reward, terminated, next_state)

        Qagent.decay_epsilon()
        done = terminated or truncated
        state = next_state
envQ.close()

# simulation_results = pd.DataFrame({"run": run_history})
# train SARSA
envS = RecordEpisodeStatistics(env, deque_size=n_episodes)

for episode in trange(n_episodes):
    state, info = envS.reset()
    done = False

    while not done:
        action = Sagent.epsilon_greedy(state)
        next_state, reward, terminated, truncated, info = envS.step(action)
        Sagent.learn(state, action, reward, terminated, next_state)

        Sagent.decay_epsilon()
        done = terminated or truncated
        state = next_state
envS.close()


# plotting


# plot training performance
returns = envQ.return_queue
plt.plot(returns)
plt.xlabel("Episodes (Q-Learning agent)")
plt.ylabel("Returns")
plt.show()

returns = envS.return_queue
plt.plot(returns)
plt.xlabel("Episodes (SARSA agent)")
plt.ylabel("Returns")
plt.show()

# plot training error
plt.plot(Qagent.training_error)
plt.xlabel("Episodes (Q-Learning  agent)")
plt.ylabel("Training Error")
plt.show()

plt.plot(Sagent.training_error)
plt.xlabel("Episodes (SARSA  agent)")
plt.ylabel("Training Error")
plt.show()

# evaluate agent
env = gym.make("CliffWalking-v0", render_mode="human")
state, _ = env.reset()
done = False
while not done:
    action = Qagent.greedy(state)
    next_state, _, terminated, truncated, _ = env.step(action)

    done = terminated or truncated
    state = next_state
env.close()


env = gym.make("CliffWalking-v0", render_mode="human")
state, _ = env.reset()
done = False
while not done:
    action = Sagent.greedy(state)
    next_state, _, terminated, truncated, _ = env.step(action)

    done = terminated or truncated
    state = next_state
env.close()
