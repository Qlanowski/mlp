import numpy as np
import pandas as pd


class MLP:

    def __init__(self, network_size, is_bias, activation_functions, cost_function, visualizer=None):
        self.network_size = np.array(network_size)
        self.is_bias = is_bias
        self.activation_functions = activation_functions
        self.cost_function = cost_function
        self.visualizer = visualizer

    def train(self, x, y, iterations, batch_size, learning_rate, momentum, seed=None):
        self.__init_weights(seed)
        x_train = x.transpose().to_numpy()
        y_train = y.transpose().to_numpy()
        data = self.__transform_data_to_tuples(x_train, y_train)
        for i in range(iterations):
            batches = self.__split_to_batches(data, batch_size)
            old_w_change = [np.zeros(w.shape) for w in self.weights]
            old_b_change = np.zeros(len(self.biases))
            for batch in batches:
                old_w_change, old_b_change = self.__train_with_single_batch(
                    batch,
                    learning_rate,
                    momentum,
                    old_w_change,
                    old_b_change)
                self.visualizer.update(self.weights, self.biases)
            print("iteration " + str(i+1) + " of " + str(iterations))

    def predict(self, data):
        result = data.copy().transpose().to_numpy()
        for w, b, f in zip(self.weights, self.biases, self.activation_functions):
            result = f.function(np.dot(w, result) + b)
        return pd.DataFrame(result.transpose())

    def __init_weights(self, seed):
        np.random.seed(seed)
        self.weights = [
            np.random.randn(y, x)
            for x, y
            in zip(self.network_size[:-1], self.network_size[1:])
        ]
        b_count = len(self.network_size) - 1
        self.biases = list(np.random.randn(b_count) if self.is_bias else np.zeros(b_count))

    def __train_with_single_batch(self, batch, learning_rate, momentum, old_w_change, old_b_change):
        cd_to_weights_sum = [np.zeros(w.shape) for w in self.weights]
        cd_to_bias_sum = list(np.zeros(len(self.biases)))
        for x, y in batch:
            cd_to_weights_list, cd_to_bias_list = self.__get_back_propagation(x, y)
            cd_to_weights_sum = self.__add_to_weights(cd_to_weights_sum, cd_to_weights_list)
            cd_to_bias_sum = self.__add_to_weights(cd_to_bias_sum, cd_to_bias_list)
        batch_len = len(batch)
        weights_change = self.__get_weights_change(cd_to_weights_sum, batch_len, old_w_change, learning_rate, momentum)
        bias_change = self.__get_weights_change(cd_to_bias_sum, batch_len, old_b_change, learning_rate, momentum)
        self.weights = self.__subtract_from_weights(self.weights, weights_change)
        self.biases = self.__subtract_from_weights(self.biases, bias_change)
        return weights_change, bias_change

    def __get_back_propagation(self, x, y):
        layer_inputs, activations = self.__get_values_on_layers(x)
        cd_to_activation = self.cost_function.derivative(activations[-1], y)
        cd_to_weights_list = [np.zeros(w.shape) for w in self.weights]
        cd_to_bias_list = list(np.zeros(len(self.biases)))
        for i in range(1, len(self.network_size)):
            cd_to_layer_input = self.__get_cd_to_layer_input(
                cd_to_activation,
                layer_inputs[-i],
                self.activation_functions[-i]
            )
            cd_to_weights_list[-i] = self.__get_cd_to_weights(activations[-i-1], cd_to_layer_input)
            cd_to_bias_list[-i] = self.__get_cd_to_bias(cd_to_layer_input)
            cd_to_activation = self.__get_cd_to_activation(self.weights[-i], cd_to_layer_input)
        return cd_to_weights_list, cd_to_bias_list

    def __get_values_on_layers(self, x):
        activations = [x]
        layer_inputs = []
        for w, b, f in zip(self.weights, self.biases, self.activation_functions):
            layer_inputs.append(np.dot(w, activations[-1]) + b)
            activations.append(f.function(layer_inputs[-1]))
        return layer_inputs, activations

    @staticmethod
    def __get_cd_to_layer_input(cd_to_activation, layer_input, activation_function):
        return cd_to_activation * activation_function.derivative(layer_input)

    @staticmethod
    def __add_to_weights(weights1, weights2):
        return [
            w1 + w2
            for w1, w2 in zip(weights1, weights2)
        ]

    @staticmethod
    def __subtract_from_weights(weights1, weights2):
        return [
            w1 - w2
            for w1, w2 in zip(weights1, weights2)
        ]

    @staticmethod
    def __get_weights_change(cd_to_weights_sum, count, old_weights_change, learning_rate, momentum):
        return [
            w / count * learning_rate + momentum * dw
            for w, dw in zip(cd_to_weights_sum, old_weights_change)
        ]

    @staticmethod
    def __get_cd_to_weights(activation, cd_to_layer_input):
        return np.dot(cd_to_layer_input, activation.transpose())

    def __get_cd_to_bias(self, cd_to_layer_input):
        return np.sum(cd_to_layer_input) if self.is_bias else 0

    @staticmethod
    def __get_cd_to_activation(weights, cd_to_layer_input):
        return np.dot(weights.transpose(), cd_to_layer_input)

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
