from train.activationFunction import ActivationFunction
import numpy as np


class ReLU(ActivationFunction):
    def __init__(self):
        pass

    def function(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0) * 1.
