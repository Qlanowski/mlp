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
        data = self.__transform_data_to_tuples(x_train, y_train)
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

    def __get_values_on_layers(self, x):
        activations = [x]
        layer_inputs = []
        for w, b in zip(self.weights, self.biases):
            layer_inputs.append(np.dot(w, activations[-1]) + b)
            activations.append(self.activation_function.function(layer_inputs[-1]))
        return layer_inputs, activations

    def __get_cd_to_layer_input(self, cd_to_activation, layer_input):
        return cd_to_activation * self.activation_function.derivative(layer_input)

    @staticmethod
    def __get_cd_to_weights(activation, cd_to_layer_input):
        return np.dot(cd_to_layer_input, activation.transpose())

    def __get_cd_to_bias(self, cd_to_layer_input):
        return np.sum(cd_to_layer_input) if self.is_bias else 0

    @staticmethod
    def __get_cd_to_activation(self, weights, cd_to_layer_input):
        return np.dot(weights.transpose(), cd_to_layer_input)


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

    @staticmethod
    def __get_cd_to_last_activation(activation, y):
        return 2 * (activation - y)

    @staticmethod
    def __transform_data_to_tuples(x, y):
        n = x.shape[1]
        x_vectors = np.hsplit(x, n)
        y_vectors = np.hsplit(y, n)
        return list(zip(x_vectors, y_vectors))

    @staticmethod
    def __split_to_batches(data, batch_size):
        return [
            data[i: i + batch_size]
            for i in range(0, len(data), batch_size)
        ]
