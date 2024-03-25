import torch
import random


def epsilon_greedy(q_network, env, state, epsilon, device):
    """
    Implements epsilon-greedy action selection

    Params:
    q_network: deep Q-learning network
    env: RL environment
    state: current state of env
    epsilon: float, exploration probability
    device: cpu or gpu
    """
    if random.random() < epsilon:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )
    else:
        # q_network(state) returns a tensor of shape (1, n_actions),
        # i.e. [[1, 2, ..., n_actions]]
        # tensor.max(1) returns a tuple (values, indices) where
        # values is a tensor containing the maximum value in each row
        # and indices is a tensor containing the column index of the maximum
        # tensor.max(1).indices is one dimensional and has length 1, e.g. [idx]
        # calling view (1, 1) resizes the index tensor to [[idx]]
        return q_network(state).max(1).indices.view(1, 1)


def greedy(q_network, state):X
    return q_network(state).max(1).indices.view(1, 1)
