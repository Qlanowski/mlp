import numpy as np
import pandas as pd


class MLP:

    def __init__(self, network_size, is_bias, activation_function):
        self.network_size = network_size
        self.is_bias = is_bias
        self.activation_function = activation_function

    def predict(self, data):
        result = data.copy().to_numpy()
        if self.biases:
            for w, b in zip(self.weights, self.biases):
                result = self.activation_function.function(np.dot(w, result) + b)
        else:
            for w in self.weights:
                result = self.activation_function.function(np.dot(w, result))
        return pd.DataFrame(result)

    def __init_weights(self, seed=None):
        np.random.seed(seed)
        self.weights = [
            np.random.randn(y, x)
            for x, y
            in zip(self.network_size[:-1], self.network_size[1:])
        ]
        self.biases = None if not self.is_bias else [
            np.random.randn(y, 1)
            for y
            in self.network_size[1:]
        ]
