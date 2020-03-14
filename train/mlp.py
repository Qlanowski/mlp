import numpy as np
import pandas as pd


class MLP:

    def __init__(self, network_size, is_bias, activation_function):
        self.network_size = network_size
        self.is_bias = is_bias
        self.activation_function = activation_function

    def train(self, x, y, iterations, batch_size, learning_rate, momentum):
        self.__init_weights()
        x_train = x.to_numpy()
        y_train = y.to_numpy()
        for i in range(iterations):
            batches = self.__split_to_batches(x_train, y_train, batch_size)
            for x_batch, y_batch in batches:
                self.__train_with_single_batch(x_batch, y_batch, learning_rate, momentum)

    def predict(self, data):
        result = data.copy().transpose().to_numpy()
        if self.is_bias:
            for w, b in zip(self.weights, self.biases):
                result = self.activation_function.function(np.dot(w, result) + b)
        else:
            for w in self.weights:
                result = self.activation_function.function(np.dot(w, result))
        return pd.DataFrame(result.transpose())

    def __init_weights(self, seed=None):
        np.random.seed(seed)
        self.weights = np.array([
            np.random.randn(y, x)
            for x, y
            in zip(self.network_size[:-1], self.network_size[1:])
        ])
        self.biases = None if not self.is_bias else np.random.randn(len(self.network_size) - 1)

    def __train_with_single_batch(self, x_batch, y_batch, learning_rate, momentum):
        pass

    @staticmethod
    def __split_to_batches(x, y, batch_size):
        return [
            (x[i: i + batch_size], y[i: i + batch_size])
            for i in range(0, len(x), batch_size)
        ]
