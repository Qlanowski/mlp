import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import train.parser as pr
import train.accuracy as acc
from train.mlp import MLP
from train.visualization.visualizer import Visualizer
from train.functions.sigmoid import Sigmoid


class TestConfig:
    def __init__(self, train_file, test_file, hid_layers, act_func, cost_func, is_bias, batch_size, iterations, lr, moment, seed):
        self.train_file = train_file
        self.test_file = test_file
        self.hidden_layers = hid_layers
        self.activation_function = act_func
        self.cost_function = cost_func
        self.is_bias = is_bias
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = lr
        self.momentum = moment
        self.seed = seed


def load_data(filename):
    df = pd.read_csv(filename)
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    return x, y


def get_single_classification_test_result(config):
    x, y = load_data(config.train_file)
    y, classes = pr.split_y_classes(y)
    layers = [len(x.columns)] + config.hidden_layers + [len(y.columns)]
    mlp = MLP(
        network_size=layers,
        is_bias=config.is_bias,
        activation_functions=[config.activation_function] * (len(layers) - 2) + [Sigmoid()],
        cost_function=config.cost_function,
        visualizer=Visualizer(None, None)
    )
    mlp.train(
        x,
        y,
        iterations=config.iterations,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        seed=config.seed
    )
    x_test, y_test = load_data(config.test_file)
    result = pr.merge_y_classes(mlp.predict(x_test), classes)
    accuracy = acc.get_classification_accuracy(result, y_test)
    print('accuracy:', accuracy)
    return result, accuracy


def get_classification_test_series_result(config, fields, values):
    results = []
    for val_set in values:
        for field, val in zip(fields, val_set):
            setattr(config, field, val)
        results.append(get_single_classification_test_result(config))
    return results


def save_classification_plot_to_accuracy(config, fields, values, x_label, y_label, title, filename):
    results = get_classification_test_series_result(config, fields, values)
    accuracies = [a * 100 for res, a in results]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([30, 100])
    p = ax.plot(values, accuracies, 'b')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.savefig(filename)
