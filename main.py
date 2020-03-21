from train import network
import train.parser as pr
import numpy as np

from train.cost_functions import QuadraticCostFunction
from train.functions.sigmoid import Sigmoid
from train.functions.relu import ReLU
from train.visualization.networkVisualizer import NetworkVisualizer
from train.visualization.visualizer import Visualizer


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


def run_scoring(iteration, iterations, network, train_data, test_data):
    if iteration % 500:
        return

    test_results = [(np.argmax(network.predict(x)) + 1, y) for (x, y) in test_data]
    test_result = sum(int(x == y) for (x, y) in test_results)
    test_acc = test_result / len(test_results)

    train_results = [(np.argmax(network.predict(x)) + 1, y) for (x, y) in train_data]
    train_result = sum(int(x == y) for (x, y) in train_results)
    train_acc = train_result / len(train_results)

    print(f'Iteration {iteration}/{iterations} completed; Test acc: {test_acc}; Train acc: {train_acc}')


def main():
    train_filename = r'classification\data.three_gauss.train.10000.csv'
    test_filename = r'classification\data.three_gauss.test.10000.csv'

    train_data = load_classification(train_filename)
    test_data = load_test_classification(test_filename)
    train_data_for_evaluation = load_test_classification(train_filename)

    layers = [train_data[0][0].shape[0], 10, 10, train_data[0][1].shape[0]]
    iterations = 1000
    batch_size = 10
    learning_rate = 0.1
    activation_functions = [ReLU()] * (len(layers) - 2) + [Sigmoid()]
    cost_function = QuadraticCostFunction()
    is_bias = True
    seed = 1000
    momentum = 0.01
    visualizer = Visualizer(layers, is_bias)

    net = network.Network(layers,
                          is_bias=is_bias,
                          activation_functions=activation_functions,
                          cost_function=cost_function,
                          visualizer=visualizer)
    net.train(train_data,
              iterations=iterations,
              mini_batch_size=batch_size,
              learning_rate=learning_rate,
              momentum=momentum,
              seed=seed,
              iteration_function=run_scoring,
              iteration_train_data=train_data_for_evaluation,
              iteration_test_data=test_data)


if __name__ == "__main__":
    main()
