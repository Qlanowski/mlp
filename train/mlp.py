import numpy as np
import pandas as pd


class MLP:

    def __init__(self, network_size, is_bias, activation_function):
        self.network_size = np.array(network_size)
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
        for w, b in zip(self.weights, self.biases):
            result = self.activation_function.function(np.dot(w, result) + b)
        return pd.DataFrame(result.transpose())

    def __init_weights(self, seed=None):
        np.random.seed(seed)
        self.weights = np.array([
            np.random.randn(y, x)
            for x, y
            in zip(self.network_size[:-1], self.network_size[1:])
        ])
        b_count = len(self.network_size) - 1
        self.biases = np.random.randn(b_count) if self.biases else np.zeros(b_count)

    def __train_with_single_batch(self, x_batch, y_batch, learning_rate, momentum):
        pass

    def __calculate_values_on_neutrons(self, x):
        a_array = [x]
        z_array = []
        for w, b in zip(self.weights, self.biases):
            z_array.append(np.dot(w, a_array[-1]) + b)
            a_array.append(self.activation_function.function(z_array[-1]))
        return np.array(z_array), np.array(a_array)

    def __calculate_cost_derivative_on_last_layer(self, z_array, a_array, y):
        delta = 2 * (a_array[-1] - y) * self.activation_function.derivative(z_array[-1])
        nabla_w = np.dot(delta, a_array[-2].transpose())
        if self.is_bias:
            nabla_b = delta
            return nabla_w, nabla_b
        return nabla_w

    @staticmethod
    def __split_to_batches(x, y, batch_size):
        return np.array([
            (x[i: i + batch_size], y[i: i + batch_size])
            for i in range(0, len(x), batch_size)
        ])
