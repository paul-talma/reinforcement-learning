import numpy as np
from collections import defaultdict
from IPython.display import display, Markdown, Latex, HTML
from tabulate import tabulate


class QAgent:
    """
    QAgent implements Q-Learning algorithm
    """

    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """
        Initializes a classic Q-Learning agent for the given environment.

        Params:
        env: a gymnasium RL environment
        learning_rate: how sensitive the agent is to new information
        initial_epsilon: generally set to 1.
        epsilon_decay: subtractive update to epsilon value
        final_epsilon: minimal epsilon value
        discount_factor: how much to discount future rewards
        """
        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor

        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

        self.training_error = []

    def epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def greedy(self, state):
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, terminated):
        next_q_val = (not terminated) * max(self.q_table[next_state])
        temporal_difference = (
            reward + self.discount_factor * next_q_val - self.q_table[state][action]
        )
        self.q_table[state][action] += self.learning_rate * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def state_to_coord(self, state):
        return divmod(state, 4)

    def coord_to_state(self, coord):
        return coord[0] + 4 * coord[1]

    def generate_table(self):
        return [
            [np.argmax(self.q_table[self.coord_to_state((i, j))]) for i in range(4)]
            for j in range(4)
        ]

    def print_table(self):
        table = self.generate_table()
        display(HTML(tabulate(table, tablefmt="html")))
