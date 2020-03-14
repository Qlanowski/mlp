from train.activationFunction import ActivationFunction
import numpy as np


class Sigmoid(ActivationFunction):
    def __init__(self):
        pass

    def function(self, x):
        return 1.0/(1.0+np.exp(-x))

    def derivative(self, x):
        return self.function(x)*(1-self.function(x))
