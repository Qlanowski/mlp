import numpy as np
from train.functions.activationFunction import ActivationFunction


class Identity(ActivationFunction):
    def __init__(self):
        pass

    def function(self, x):
        return x

    def derivative(self, x):
        return np.ones(x.shape)
