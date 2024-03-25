from deep_q_network import deep_q_network
import torch
import gymnasium as gym
from action_selection import greedy

env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()
state_space = len(state)
n_actions = env.action_space.n
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

q_network = deep_q_network(state_space, n_actions)
q_network.load_state_dict(
    torch.load("environments/classic_control/saved_models/cart_pole_model.pth")
)

done = False
while not done:
    # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action = greedy(q_network, state)
    next_state, _, terminated, truncated, _ = env.step(action.item())
    done = terminated or truncated
    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
    state = next_state
