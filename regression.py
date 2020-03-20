import mnist_loader
from train import network
import train.parser as pr
import numpy as np


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


train_filename = r'C:\Users\ulano\PycharmProjects\mlp\classification\data.three_gauss.test.100.csv'
test_filename = r'C:\Users\ulano\PycharmProjects\mlp\classification\data.three_gauss.test.100.csv'

train_data = load_classification(train_filename)
test_data = load_test_classification(test_filename)

layers = [train_data[0][0].shape[0], 10, 10, train_data[0][1].shape[0]]
epochs = 100
batch_size = 10
learning_rate = 0.1

net = network.Network(layers)
net.SGD(train_data, epochs=epochs, , 0.1)

test_results = [(np.argmax(net.feedforward(x))+1, y) for (x, y) in test_data]
result = sum(int(x == y) for (x, y) in test_results)
print(result / len(test_results))
net.evaluate(test_data)
