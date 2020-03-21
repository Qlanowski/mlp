from train import network
import train.parser as pr
import numpy as np

from train.cost_functions import QuadraticCostFunction
from train.functions.sigmoid import Sigmoid
from train.functions.relu import ReLU


def load_classification(filename):
    x, y = pr.load_data(filename)
    y = pr.split_y_classes(y)

    x = np.array(x)
    y = np.array(y[0])
    return [(np.array(_x).reshape(-1, 1), np.array(_y).reshape(-1, 1)) for _x, _y in zip(x, y)]


def load_test_classification(filename):
    x, y = pr.load_data(filename)
    x = np.array(x)
    y = np.array(y)
    return [(np.array(_x).reshape(-1, 1), _y[0]) for _x, _y in zip(x, y)]


train_filename = r'C:\Users\ulano\PycharmProjects\mlp\classification\data.three_gauss.train.100.csv'
test_filename = r'C:\Users\ulano\PycharmProjects\mlp\classification\data.three_gauss.test.100.csv'

train_data = load_classification(train_filename)
test_data = load_test_classification(test_filename)

layers = [train_data[0][0].shape[0], 10, 10, train_data[0][1].shape[0]]
iterations = 1000
batch_size = 10
learning_rate = 0.1
activation_functions = [ReLU()] * (len(layers) - 2) + [Sigmoid()]
cost_function = QuadraticCostFunction()
is_bias=True

net = network.Network(layers,
                      is_bias=is_bias,
                      activation_functions=activation_functions,
                      cost_function=cost_function)
net.SGD(train_data,
        iterations=iterations,
        mini_batch_size=batch_size,
        learning_rate=learning_rate)

test_results = [(np.argmax(net.feedforward(x)) + 1, y) for (x, y) in test_data]
result = sum(int(x == y) for (x, y) in test_results)
print(result / len(test_results))
