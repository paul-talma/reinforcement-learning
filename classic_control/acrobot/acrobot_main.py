import torch
import matplotlib
from matplotlib import pyplot as plt
from action_selection import epsilon_greedy
from deep_q_network import deep_q_network
from replay_buffer import ExperienceReplayBuffer
from training_functions import soft_update, optimize
import torch.optim as optim
from visualization import plot_rewards
import gymnasium as gym
import time

is_ipython = "inline" in matplotlib.get_backend()
plt.ion()
plt.style.use("Solarize_Light2")

# set up environment and parameters
env = gym.make("Acrobot-v1")
seed = 0
obs, _ = env.reset(seed=seed)
state_space = len(obs)
n_actions = env.action_space.n

# agent parameters
initial_epsilon = 0.95
epsilon_decay = 0.9999
final_epsilon = 0.05
discount_factor = 0.99

# training parameters
tau = 0.005
batch_size = 128
learning_rate = 0.0001
buffer_capacity = 10_000
gpu_available = torch.backends.mps.is_available()
if gpu_available:
    device = torch.device("mps")
    n_episodes = 400
else:
    device = torch.device("cpu")
    n_episodes = 30

# somehow, training is faster on cpu, so we use this for now
device = torch.device("cpu")

# initialize policy and target nets
policy_net = deep_q_network(state_space, n_actions).to(device)
target_net = deep_q_network(state_space, n_actions).to(device)
optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate, amsgrad=True)

# training loop
epsilon = initial_epsilon
buffer = ExperienceReplayBuffer(buffer_capacity)
episode_rewards = []
epsilon_values = [epsilon]
t0 = time.time()
steps = 0

for episode in range(n_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    episode_reward = 0
    while not done:
        action = epsilon_greedy(policy_net, env, state, epsilon, device)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        episode_reward += reward
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                next_state, dtype=torch.float32, device=device
            ).unsqueeze(0)

        buffer.add(state, action, reward, next_state)

        state = next_state
        # decay epsilon
        # epsilon = final_epsilon + (initial_epsilon - final_epsilon) * math.exp(
        #     -1.0 * steps / epsilon_decay
        # )
        # steps += 1
        epsilon = max(epsilon * epsilon_decay, final_epsilon)

        optimize(
            policy_net,
            target_net,
            buffer,
            batch_size,
            device,
            discount_factor,
            optimizer,
        )

        soft_update(policy_net, target_net, tau)

    epsilon_values.append(epsilon)
    episode_rewards.append(episode_reward)
    plot_rewards(episode_rewards, epsilon_values, is_ipython)
env.close()

print(f"Training complete. Time elapsed: {time.time() - t0}")

plot_rewards(episode_rewards, epsilon_values, is_ipython, show_results=True)
plt.ioff()
plt.show()


# save model
torch.save(
    policy_net.state_dict(),
    "/Users/paultalma/Programming/Python/reinforcement-learning/classic_control/acrobot/saved_models/acrobot_model.pth",
)
