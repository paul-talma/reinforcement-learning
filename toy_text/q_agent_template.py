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
        n_states: int,
        n_actions: int,
        discount_factor: float = 0.95,
    ):
        """
        Initializes a classic Q-Learning agent for the given environment.

        Params:
        env: a gymnasium RL environment
        learning_rate: how sensitive the agent is to new information
        initial_epsilon: generally set to 1.
        epsilon_decay: multiplicative update to epsilon value
        final_epsilon: minimal epsilon value
        n_states: number of states
        n_actions: number of actions
        discount_factor: how much to discount future rewards
        step_wise_decay: whether to decay epsilon in batches or continuously
        """
        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor

        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((self.n_states, self.n_actions))
        # self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

        self.training_error = []

    def epsilon_greedy(self, state):
        """
        Implements epsilon-greedy action selection: with probability
        epsilon, chooses a random action, otherwise chooses action
        that maximizes Q-value.
        If all actions have the same Q-value, chooses
        randomly amongst them.

        Params:
        state: state of the environment
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # explore
        else:
            # if all actions have equal value, choose randomly
            # (np.argmax would always choose first action)
            if np.all(self.q_table[state][:]) == self.q_table[state][0]:
                return self.env.action_space.sample()
            else:
                return np.argmax(self.q_table[state])  # exploit

    def greedy(self, state):
        """
        Implements greedy action selection: chooses action with highest
        Q-value. There is no need to randomize choice when actions have
        the same value, as there is no concern with exploration.

        Params:
        state: state of the environment
        """
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, terminated):
        """
        Implements Q-Learning algorithm:
        Q'(s, a) = Q(s, a) + alpha [R(s, a) + gamma (max_a' Q(s', a')) - Q(s, a)]

        Params:
        state: state of the environment
        action: current action
        reward: reward for (state, action) transition
        next_state: result of taking action in state
        terminated: whether (s, a) lands agent in terminal state
        """
        next_q_val = (not terminated) * max(self.q_table[next_state])
        temporal_difference = (
            reward + self.discount_factor * next_q_val - self.q_table[state][action]
        )
        self.q_table[state][action] += self.learning_rate * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """
        Reduces epsilon value over time.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def reset_q_values(self):
        self.q_table = np.zeros((self.n_states, self.n_actions))
