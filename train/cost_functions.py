import numpy as np


class CostFunction:

    def function(self, activation, y):
        pass

    def derivative(self, activation, y):
        pass


class QuadraticCostFunction(CostFunction):

    def function(self, activation, y):
        return ((activation - y)/2) ** 2

    def derivative(self, activation, y):
        return activation - y


class CrossEntropyCostFunction(CostFunction):

    def function(self, activation, y):
        return y * np.log(activation) + (1 - y) * np.log(1 - activation)

    def derivative(self, activation, y):
        return (activation - y) / ((1 - activation) * activation)
