import numpy as np
from collections import defaultdict
class BlackjackAgent:
    def __init__(
            self,
            epsilon_decay,
            final_epsilon,
            alpha=0.2,
            initial_epsilon=0.2,
            gamma=0.95
            ):
        """
        Initialize an RL agent
        :param learning_rate:
        :param initial_epsilon:
        :param epsilon_decay:
        :param final_epsilon:
        :param gamma:
        """
        self.q_values = defaultdict(lambda: np.zeroes(env.action_space.n))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Implements epsilon-greedy action selection
        :param obs: tuple (player's sum, dealer's face-up card, usable ace)
        :return: action
        """
        if np.random.random() < self.epsilon: # epsilon explore
            return env.action_space.sample()
        else: # greedy exploit
            return int(np.argmax(self.q_values[obs]))

    def update(self,
               obs: tuple[int, int, bool],
               action: int,
               reward: float,
               terminated: bool,
               next_obs: tuple[int, int, bool]
               ):
        """
        Updates Q-value of an action
        :param obs: current state
        :param action: current action
        :param reward: reward for performing action in current state
        :param terminated: did action end episode?
        :param next_obs: resulting state from (state, action) pair
        """
        future_q_val = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = reward + self.gamma * future_q_val - self.q_values[obs][action]

        # update Q-values
        self.q_values[obs][action] += self.alpha * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)














