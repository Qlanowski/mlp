from train.functions.activationFunction import ActivationFunction
import numpy as np


class Tanh(ActivationFunction):
    def __init__(self):
        pass

    def function(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1.0 - np.tanh(x) ** 2
