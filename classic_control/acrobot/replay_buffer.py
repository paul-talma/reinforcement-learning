from collections import namedtuple, deque
import random

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


class ExperienceReplayBuffer:
    def __init__(self, capacity):
        """
        Creates an experience replay buffer with length `capacity`
        Implemented using a `deque`

        Params:
        capacity: int
        """
        self.buffer = deque([], maxlen=capacity)

    def add(self, *args):
        """
        Adds the experience specified by the `args` to the experience buffer
        `args` are first stored as a `Transition`

        Params:
        args: should be state, action, reward, next state
        """
        self.buffer.append(Transition(*args))

    def sample_experience(self, batch_size):
        """
        Sample `batch_size` experiences from buffer. Returns a list of `Transitions`

        Params:
        batch_size: int
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """
        More ergonomic way of getting the length of the buffer
        """
        return len(self.buffer)
