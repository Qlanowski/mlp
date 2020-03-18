import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

import train.parser as pr
import train.scores as sc
from train.mlp import MLP
from train.visualization.visualizer import Visualizer
from train.functions.sigmoid import Sigmoid


class TestConfig:
    def __init__(self, train_file, test_file, hid_layers, act_func, cost_func, is_bias, batch_size, lr, moment, seed):
        self.train_file = train_file
        self.test_file = test_file
        self.hidden_layers = hid_layers
        self.activation_function = act_func
        self.cost_function = cost_func
        self.is_bias = is_bias
        self.batch_size = batch_size
        self.learning_rate = lr
        self.momentum = moment
        self.seed = seed


class TestResult:
    def __init__(self, iterations, test_set_result, test_set_score, train_set_result, train_set_score):
        self.iterations = iterations
        self.test_set_result = test_set_result
        self.test_set_score = test_set_score
        self.train_set_result = train_set_result
        self.train_set_score = train_set_score


def get_test_filename(config):
    return (f'{type(config.activation_function).__name__.lower()}_'
            f'{"-".join(str(l) for l in config.hidden_layers)}l_'
            f'{ "" if config.is_bias else "n"}b_'
            f'{type(config.cost_function).__name__.lower()}_'
            f'{config.batch_size}bs_'
            f'{config.learning_rate}lr_'
            f'{config.momentum}m_'
            f'{config.seed}s_'
            f'{os.path.basename(os.path.splitext(config.train_file)[0])}.png')


def get_test_mlp(config, input_size, output_size, activation_functions):
    layers = [input_size] + config.hidden_layers + [output_size]
    return MLP(
        network_size=layers,
        is_bias=config.is_bias,
        activation_functions=activation_functions,
        cost_function=config.cost_function,
        visualizer=Visualizer(None, None)
    )


def get_single_classification_test_result(config, iterations):
    x_train, y = pr.load_data(config.train_file)
    y_train, classes = pr.split_y_classes(y)
    activation_functions = [config.activation_function] * len(config.hidden_layers) + [Sigmoid()]
    mlp = get_test_mlp(config, len(x_train.columns), len(y_train.columns), activation_functions)
    mlp.train(
        x_train,
        y_train,
        iterations=iterations,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        seed=config.seed
    )
    x_test, y_test = pr.load_data(config.test_file)
    test_set_result = pr.merge_y_classes(mlp.predict(x_test), classes)
    train_set_result = pr.merge_y_classes(mlp.predict(x_train), classes)
    test_set_accuracy = sc.get_classification_accuracy(test_set_result, y_test)
    train_set_accuracy = sc.get_classification_accuracy(train_set_result, y)
    print('test set accuracy:', test_set_accuracy)
    print('train set accuracy:', train_set_accuracy)
    return TestResult(
        iterations,
        test_set_result,
        test_set_accuracy,
        train_set_result,
        train_set_accuracy
    )


def get_classification_test_series_result(config, iterations_list):
    results = []
    for iterations in iterations_list:
        results.append(get_single_classification_test_result(config, iterations))
    return results


def visualise_classification_test_series(results, title, save=False, filename=None):
    train_accuracies = [res.train_set_score * 100 for res in results]
    test_accuracies = [res.test_set_score * 100 for res in results]
    iterations = [res.iterations for res in results]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 100, 5))
    ax.grid(True)
    _ = ax.plot(iterations, train_accuracies, 'r', label='training data')
    _ = ax.plot(iterations, test_accuracies, 'b', label='test data')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Accuracy [%]')
    ax.set_title(title)
    ax.legend(loc='lower right')
    if save:
        fig.savefig(filename)
    else:
        fig.show()
