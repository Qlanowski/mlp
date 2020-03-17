from train.functions.activationFunction import ActivationFunction
import numpy as np


class Logistic(ActivationFunction):
    def __init__(self):
        pass

    def function(self, x):
        return 1/(1 + np.exp(-x))

    def derivative(self, x):
        return self.function(x)*(1-self.function(x))
