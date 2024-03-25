import torch.nn as nn
import torch.nn.functional as F


class deep_q_network(nn.Module):
    def __init__(self, state_space, n_actions):
        super(deep_q_network, self).__init__()
        self.layer1 = nn.Linear(state_space, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, input):
        output = F.relu(self.layer1(input))
        output = F.relu(self.layer2(output))
        return self.layer3(output)
