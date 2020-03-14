import numpy as np


class MLP:

    def __init__(self, network_size, is_bias, activation_function):
        self.weights = [
            np.ones(y, x)
            for x, y
            in zip(network_size[:-1], network_size[1:])
        ]
        self.biases = None if not is_bias else [
            np.ones(y, 1)
            for y
            in network_size[1:]
        ]
        self.activation_function = activation_function
