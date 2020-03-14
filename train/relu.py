from train.activationFunction import ActivationFunction
import numpy as np


class ReLU(ActivationFunction):
    def __init__(self):
        pass

    def function(self, z):
        return np.maximum(0, z)

    def derivative(self, z):
        return (z > 0) * 1.
