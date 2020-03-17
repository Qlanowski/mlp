from train.functions.activationFunction import ActivationFunction
import numpy as np


class ReLU(ActivationFunction):
    def __init__(self):
        pass

    def function(self, x):
        output = np.where(x > 0, x, x * 0.01)
        return output

    def derivative(self, x):
        dx = np.ones_like(x)
        dx[x < 0] = 0.01
        return dx
