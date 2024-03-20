import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def q_network(input_shape, n_actions):
    model = Sequential(
        [
            Dense(128, activation="relu", input_shape=input_shape),
            Dense(128, activation="relu"),
            Dense(n_actions, activation="linear"),
        ]
    )
    model.compile(loss="mse", optimizer=Adam(lr=0.001))
