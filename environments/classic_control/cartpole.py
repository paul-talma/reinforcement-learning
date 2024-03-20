import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from deep_q_network import q_network
from experience_replay import ExperienceReplayBuffer
import params as params

from tqdm import trange

env = gym.make("CartPole-v1")


params = params.CartPoleParams(
    # framework params
    n_episodes=10_000,
    n_runs=1,
    roll_window=100,
    # agent params
    learning_rate=0.8,
    discount_factor=0.98,
    initial_epsilon=1,
    epsilon_decay=0.998,
    final_epsilon=0.01,
    # env params
    seed=100,
    n_actions=env.action_space.n,
    n_states=env.observation_space.n,
    state_space_shape=env.observation_space.shape,
    buffer_capacity=10000,
    batch_size=32,
)


def epsilon_greedy(q_vals, epsilon):
    n_actions = len(q_vals)
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(q_vals)


def train_network(
    env,
    q_network,
    buffer,
    n_episodes,
    batch_size,
    discount_factor,
    initial_epsilon,
    epsilon_decay,
    final_epsilon,
):
    epsilon = initial_epsilon
    for episode in trange(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            q_vals = q_network.predict(np.array([state]))[0]
            action = epsilon_greedy(q_vals, action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            buffer.add(state, action, reward, next_state, done)
            state = next_state

            if buffer.size() >= batch_size:
                mini_batch = buffer.sample(batch_size)
                for s, a, r, next_s, d in mini_batch:
                    target = (not d) * r + discount_factor * np.max(
                        q_network.predict(np.array([next_s]))[0]
                    )
                    target_q_vals = q_network.predict(np.array([s]))
                    target_q_vals[0][a] = target
                    q_network.fit(np.array([s]), target_q_vals, verbose=False)

            if epsilon > final_epsilon:
                epsilon *= epsilon_decay

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")


q_network = q_network(params.state_space_shape, params.n_actions)
buffer = ExperienceReplayBuffer(params.buffer_capacity)
train_network(
    env=env,
    q_network=q_network,
    buffer=buffer,
    n_episodes=params.n_episodes,
    batch_size=params.batch_size,
    discount_factor=params.discount_factor,
    initial_epsilon=params.initial_epsilon,
    epsilon_decay=params.epsilon_decay,
    final_epsilon=params.final_epsilon,
)
