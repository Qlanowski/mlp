class CostFunction:

    def function(self, activation, y):
        pass

    def derivative(self, activation, y):
        pass


class QuadraticCostFunction(CostFunction):

    def function(self, activation, y):
        return (activation - y) ** 2

    def derivative(self, activation, y):
        return 2 * (activation - y)