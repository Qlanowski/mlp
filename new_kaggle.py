from train import network
import train.parser as pr
import numpy as np

from train.cost_functions import QuadraticCostFunction
from train.functions.sigmoid import Sigmoid
from train.functions.relu import ReLU

from train.visualization.visualizer import Visualizer

import mnist_loader


def run_scoring(iteration, iterations, network, train_data, test_data):
    if iteration % 500:
        return

    test_results = [(np.argmax(network.predict(x)), y) for (x, y) in test_data]
    result = sum(int(x == y) for (x, y) in test_results)
    test_acc = result / len(test_results)

    print(f'Iteration {iteration}/{iterations} completed; Test acc: {test_acc}')

def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)

    layers = [training_data[0][0].shape[0], 10, training_data[0][1].shape[0]]
    batch_size = 10
    iterations = len(training_data) / batch_size * 10
    learning_rate = 3
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
    net.train(training_data,
              iterations=iterations,
              mini_batch_size=batch_size,
              learning_rate=learning_rate,
              momentum=momentum,
              seed=seed,
              iteration_function=run_scoring,
              iteration_train_data=[],
              iteration_test_data=list(test_data))


if __name__ == "__main__":
    main()