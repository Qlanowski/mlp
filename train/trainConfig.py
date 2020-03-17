import random

from train.functions.identity import Identity
from train.functions.logistic import Logistic
from train.functions.relu import ReLU
from train.functions.sigmoid import Sigmoid
from train.functions.tanh import Tanh
from train.visualization.networkVisualizer import NetworkVisualizer
from train.visualization.visualizer import Visualizer


class TrainConfig:
    seed_range = 1000

    def __init__(self, layers, activation_function, bias, batch_size, number_of_iterations, learning_rate, momentum,
                 problem, input_file, test_file, seed, visualizer):
        self.layers = layers
        if activation_function == 0:
            self.activation_function = ReLU()
        elif activation_function == 1:
            self.activation_function = Sigmoid()
        elif activation_function == 2:
            self.activation_function = Tanh()
        elif activation_function == 3:
            self.activation_function = Logistic()
        elif activation_function == 4:
            self.activation_function = Identity()

        self.bias = bias == 1
        self.batch_size = batch_size
        self.number_of_iterations = number_of_iterations
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.problem = problem
        self.input_file = input_file
        self.test_file = test_file
        if int(seed) == -1:
            self.seed = int(random.random() * self.seed_range)
        else:
            self.seed = seed

        if int(visualizer) == 1:
            self.visualizer = NetworkVisualizer(self.layers, self.bias)
        else:
            self.visualizer = Visualizer(self.layers, self.bias)
