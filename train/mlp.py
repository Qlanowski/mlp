import numpy as np
import pandas as pd


class MLP:

    def __init__(self, network_size, is_bias, activation_function):
        self.network_size = np.array(network_size)
        self.is_bias = is_bias
        self.activation_function = activation_function

    def train(self, x, y, iterations, batch_size, learning_rate, momentum):
        self.__init_weights()
        x_train = x.transpose().to_numpy()
        y_train = y.transpose().to_numpy()
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
        self.weights = [
            np.random.randn(y, x)
            for x, y
            in zip(self.network_size[:-1], self.network_size[1:])
        ]
        b_count = len(self.network_size) - 1
        self.biases = list(np.random.randn(b_count) if self.is_bias else np.zeros(b_count))

    def __train_with_single_batch(self, x_batch, y_batch, learning_rate, momentum):
        nablas = self.back_propagation(x_batch, y_batch)
        if self.is_bias:
            nabla_w, nabla_b = nablas
            self.biases = self.biases - nabla_b * learning_rate
        else:
            nabla_w = nablas
        self.weights = self.weights - nabla_w * learning_rate

    def __calculate_values_on_neutrons(self, x):
        a_array = [x.transpose()]
        z_array = []
        for w, b in zip(self.weights, self.biases):
            z_array.append(np.dot(w, a_array[-1]) + b)
            a_array.append(self.activation_function.function(z_array[-1]))
        return np.array(z_array), np.array(a_array)

    def __calculate_cost_derivative_on_last_layer(self, z_array, a_array, y):
        delta = 2 * (a_array[-1] - y.transpose()) * self.activation_function.derivative(z_array[-1])
        nabla_w = np.dot(delta, a_array[-2].transpose())
        if self.is_bias:
            nabla_b = delta
            return delta, nabla_w, nabla_b
        return delta, nabla_w

    def __calculate_cost_derivative_on_prev_layer(self, z, a, weights, next_delta):
        delta = np.dot(weights.transpose(), next_delta) * self.activation_function.derivative(z)
        nabla_w = np.dot(delta, a.transpose())
        if self.is_bias:
            nabla_b = delta
            return delta, nabla_w, nabla_b
        return delta, nabla_w

    def __back_propagation(self, x, y):
        z_array, a_array = self.__calculate_values_on_neutrons(x)
        nabla_w = []
        nabla_b = []
        delta, *nablas = self.__calculate_cost_derivative_on_last_layer(z_array, a_array, y)
        nabla_w.append(nablas[0])
        if self.is_bias:
            nabla_b.append(nablas[1])
        for i in range(2, len(self.network_size)):
            delta, *nablas = self.__calculate_cost_derivative_on_prev_layer(
                z_array[-i],
                a_array[-i - 1],
                self.weights[-i + 1],
                delta
            )
            nabla_w.append(nablas[0])
            if self.is_bias:
                nabla_b.append(nablas[1])
        nabla_w.reverse()
        if self.is_bias:
            nabla_b.reverse()
            return np.array(nabla_w), np.array(nabla_b)
        return np.array(nabla_w)

    def __get_cost_derivative_with_respect_to_network_output(self, a, y):
        return 2 * np.sum(a - y, axis=1)

    @staticmethod
    def __transform_data_to_tuples(x, y):
        n = x.shape[1]
        x_vectors = np.hsplit(x, n)
        y_vectors = np.hsplit(y, n)
        return list(zip(x_vectors, y_vectors))

    @staticmethod
    def __split_to_batches(x, y, batch_size):
        n = x.shape[1]
        batch_count = n // batch_size
        rest = n % batch_size
        x_batches = list(np.hsplit(x[:, np.r_[:n - rest]], batch_count))
        y_batches = list(np.hsplit(y[:, np.r_[:n - rest]], batch_count))
        if rest > 0:
            x_batches.append(x[:, np.r_[n - rest:n]])
            y_batches.append(y[:, np.r_[n - rest:n]])
        return list(zip(x_batches, y_batches))
