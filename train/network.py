import random
import numpy as np


class Network(object):

    def __init__(self, sizes, is_bias, activation_functions, cost_function, visualizer):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.is_bias = is_bias
        self.activation_functions = activation_functions
        self.cost_function = cost_function
        self.visualizer = visualizer

    def predict(self, a):
        for b, w, f in zip(self.biases, self.weights, self.activation_functions):
            a = f.function(np.dot(w, a) + b)
        return a

    def train(self, training_data, iterations, mini_batch_size, learning_rate, momentum, seed,
              iteration_function=None,
              iteration_train_data=None,
              iteration_test_data=None):
        self.__init_weights(seed)
        random.seed(seed)

        training_data = list(training_data)
        n = len(training_data)

        i = 0
        while i < iterations:
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate, momentum)
                self.visualizer.update(self.weights, self.biases)
                if iteration_function:
                    iteration_function(i+1, iterations, self, iteration_train_data, iteration_test_data)
                i += 1
                if i >= iterations:
                    return

    def update_mini_batch(self, mini_batch, learning_rate, momentum):
        nabla_b = np.zeros(len(self.biases))
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            if (self.is_bias):
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.last_weights_change = [(learning_rate / len(mini_batch)) * nw + momentum * lc
                                    for lc, nw in zip(self.last_weights_change, nabla_w)]
        self.weights = [w - lc for w, lc in zip(self.weights, self.last_weights_change)]

        self.last_biases_change = [(learning_rate / len(mini_batch)) * nb + momentum * lc
                                   for lc, nb in zip(self.last_biases_change, nabla_b)]
        self.biases = [b - lc for b, lc in zip(self.biases, self.last_biases_change)]

    def backprop(self, x, y):
        nabla_b = np.zeros(len(self.biases))
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feed_forward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w, f in zip(self.biases, self.weights, self.activation_functions):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = f.function(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                self.activation_functions[-1].derivative(zs[-1])
        nabla_b[-1] = np.sum(delta)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_functions[-l].derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = np.sum(delta)
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        return self.cost_function.derivative(output_activations, y)

    def __init_weights(self, seed):
        np.random.seed(seed)
        b_count = len(self.sizes) - 1
        self.biases = np.random.randn(b_count) if self.is_bias else np.zeros(b_count)
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.last_biases_change = np.zeros(self.biases.shape)
        self.last_weights_change = [np.zeros(w.shape) for w in self.weights]
