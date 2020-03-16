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
            batches = self.__split_to_batches(data, batch_size)
            for batch in batches:
                self.__train_with_single_batch(batch, learning_rate, momentum)

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

    def __train_with_single_batch(self, batch, learning_rate, momentum):
        cd_to_weights_sum = [np.zeros(w.shape) for w in self.weights]
        cd_to_bias_sum = list(np.zeros(len(self.biases)))
        for x, y in batch:
            cd_to_weights_list, cd_to_bias_list = self.__get_back_propagation(x, y)
            cd_to_weights_sum = [
                w + w1
                for w, w1 in zip(cd_to_weights_sum, cd_to_weights_list)
            ]
            cd_to_bias_sum = [
                b + b1
                for b, b1 in zip(cd_to_bias_sum, cd_to_bias_list)
            ]
        batch_len = len(batch)
        self.weights = [
            w - cd_w / batch_len * learning_rate
            for w, cd_w in zip(self.weights, cd_to_weights_sum)
        ]
        self.biases = [
            b - cd_b / batch_len * learning_rate
            for b, cd_b in zip(self.biases, cd_to_bias_sum)
        ]

    def __get_back_propagation(self, x, y):
        layer_inputs, activations = self.__get_values_on_layers(x)
        cd_to_activation = self.__get_cd_to_last_activation(activations[-1], y)
        cd_to_weights_list = [np.zeros(w.shape) for w in self.weights]
        cd_to_bias_list = list(np.zeros(len(self.biases)))
        for i in range(1, len(self.network_size)):
            cd_to_layer_input = self.__get_cd_to_layer_input(cd_to_activation, layer_inputs[-i])
            cd_to_weights_list[-i] = self.__get_cd_to_weights(activations[-i-1], cd_to_layer_input)
            cd_to_bias_list[-i] = self.__get_cd_to_bias(cd_to_layer_input)
            cd_to_activation = self.__get_cd_to_activation(self.weights[-i], cd_to_layer_input)
        return cd_to_weights_list, cd_to_bias_list

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
    def __get_cd_to_activation(weights, cd_to_layer_input):
        return np.dot(weights.transpose(), cd_to_layer_input)

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
