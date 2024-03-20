import numpy as np


class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer, size=batch_size, replace=False))
        return (self.buffer[index] for index in indices)

    def size(self):
        return len(self.buffer)
