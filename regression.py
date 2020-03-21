from train import network
import train.parser as pr
import numpy as np

from train.cost_functions import QuadraticCostFunction
from train.functions.identity import Identity
from train.functions.relu import ReLU
from train.functions.sigmoid import Sigmoid


def load_regression(filename):
    x, y = pr.load_data(filename)
    x = np.array(x)
    y = np.array(y)
    return [(np.array(_x).reshape(-1, 1), np.array(_y).reshape(-1, 1)) for _x, _y in zip(x, y)]


def load_test_regression(filename):
    x, y = pr.load_data(filename)
    x = np.array(x)
    y = np.array(y)
    return [(np.array(_x).reshape(-1, 1), _y[0]) for _x, _y in zip(x, y)]


train_filename = r'C:\Users\ulano\PycharmProjects\mlp\regression\data.activation.train.100.csv'
test_filename = r'C:\Users\ulano\PycharmProjects\mlp\regression\data.activation.test.100.csv'

train_data = load_regression(train_filename)
test_data = load_test_regression(test_filename)

layers = [train_data[0][0].shape[0], 7, train_data[0][1].shape[0]]
iterations = 3000
batch_size = 10
learning_rate = 0.0001
activation_functions = [ReLU()] * (len(layers) - 2) + [Identity()]
cost_function = QuadraticCostFunction()
is_bias = True
seed = 1000
momentum=0.01,

net = network.Network(layers,
                      is_bias=is_bias,
                      activation_functions=activation_functions,
                      cost_function=cost_function)
net.SGD(train_data,
        iterations=iterations,
        mini_batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=momentum,
        seed=seed)

test_results = [(net.feedforward(x), y) for (x, y) in test_data]
npa = np.array(test_results)
result = sum((x - y) ** 2 for (x, y) in test_results)
print(f'Test score: {result}')

train_results = [(net.feedforward(x), y) for (x, y) in train_data]
result2 = sum((x - y) ** 2 for (x, y) in train_results)
print(f'Train score: {result2}')
