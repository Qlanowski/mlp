import os

from train.cost_functions import QuadraticCostFunction
from train.functions.identity import Identity
from train.functions.relu import ReLU
from train.functions.sigmoid import Sigmoid
from train.functions.tanh import Tanh
import train.parser as pr
import numpy as np
import tests.test_visualisation as tv
from train.network import Network
from train.visualization.visualizer import Visualizer


class TestResult:
    def __init__(self, iterations, test_score, train_score):
        self.iterations = iterations
        self.test_score = test_score
        self.train_score = train_score


class ClassificationConfiguration:
    def __init__(self, train_file, test_file, iterations, hid_layers, act_func,
                 cost_func, is_bias, batch_size, lr, moment, seed, collect_results_for_iterations):
        self.train_file = train_file
        self.test_file = test_file
        self.iterations = iterations
        self.hidden_layers = hid_layers
        self.activation_function = act_func
        self.cost_function = cost_func
        self.is_bias = is_bias
        self.batch_size = batch_size
        self.learning_rate = lr
        self.momentum = moment
        self.seed = seed
        self.collect_results_for_iterations = collect_results_for_iterations
        self.results = []

    def get_test_title(self):
        return f'Classification for {type(self.activation_function).__name__}'

    def get_score_name(self):
        return "Accuracy %"

    def get_name(self):
        return (f'{type(self.activation_function).__name__.lower()}_'
                f'{"-".join(str(l) for l in self.hidden_layers)}l_'
                f'{"" if self.is_bias else "n"}b_'
                f'{type(self.cost_function).__name__.lower()}_'
                f'{self.batch_size}bs_'
                f'{self.learning_rate}lr_'
                f'{self.momentum}m_'
                f'{self.seed}s_'
                f'{os.path.basename(os.path.splitext(self.train_file)[0])}')

    def perform_test(self):
        train_data = self.__get_train_data(self.train_file)
        test_data = self.__get_test_data(self.test_file)
        train_data_for_evaluation = self.__get_test_data(self.train_file)

        layers = self.hidden_layers + [train_data[0][1].shape[0]]
        layers.insert(0, train_data[0][0].shape[0])
        activation_functions = [self.activation_function] * (len(layers) - 2) + [Sigmoid()]
        visualizer = Visualizer(layers, self.is_bias)
        net = Network(layers,
                      is_bias=self.is_bias,
                      activation_functions=activation_functions,
                      cost_function=self.cost_function,
                      visualizer=visualizer)
        net.train(train_data,
                  iterations=self.iterations,
                  mini_batch_size=self.batch_size,
                  learning_rate=self.learning_rate,
                  momentum=self.momentum,
                  seed=self.seed,
                  iteration_function=self.__run_scoring,
                  iteration_train_data=train_data_for_evaluation,
                  iteration_test_data=test_data)

        return self.results

    @staticmethod
    def __get_train_data(filename):
        x, y = pr.load_data(filename)
        y = pr.split_y_classes(y)
        x = np.array(x)
        y = np.array(y[0])
        return [(np.array(_x).reshape(-1, 1), np.array(_y).reshape(-1, 1)) for _x, _y in zip(x, y)]

    @staticmethod
    def __get_test_data(filename):
        x, y = pr.load_data(filename)
        x = np.array(x)
        y = np.array(y)
        return [(np.array(_x).reshape(-1, 1), _y[0]) for _x, _y in zip(x, y)]

    def __run_scoring(self, iteration, iterations, network, train_data, test_data):
        if iteration % self.collect_results_for_iterations:
            return

        test_results = [(np.argmax(network.predict(x)) + 1, y) for (x, y) in test_data]
        test_result = sum(int(x == y) for (x, y) in test_results)
        test_acc = test_result / len(test_results)

        train_results = [(np.argmax(network.predict(x)) + 1, y) for (x, y) in train_data]
        train_result = sum(int(x == y) for (x, y) in train_results)
        train_acc = train_result / len(train_results)
        self.results += [TestResult(iteration, test_acc * 100, train_acc * 100)]
        print(f'Iteration {iteration}/{iterations} completed; Test acc: {test_acc:.4f}; Train acc: {train_acc:.4f}')


class RegressionConfiguration:
    def __init__(self, train_file, test_file, iterations, hid_layers, act_func,
                 cost_func, is_bias, batch_size, lr, moment, seed, collect_results_for_iterations):
        self.train_file = train_file
        self.test_file = test_file
        self.iterations = iterations
        self.hidden_layers = hid_layers
        self.activation_function = act_func
        self.cost_function = cost_func
        self.is_bias = is_bias
        self.batch_size = batch_size
        self.learning_rate = lr
        self.momentum = moment
        self.seed = seed
        self.collect_results_for_iterations = collect_results_for_iterations
        self.results = []

    def get_test_title(self):
        return f'Regression for {type(self.activation_function).__name__}'

    def get_score_name(self):
        return f"Error ({type(self.cost_function).__name__})"

    def get_name(self):
        return (f'{type(self.activation_function).__name__.lower()}_'
                f'{"-".join(str(l) for l in self.hidden_layers)}l_'
                f'{"" if self.is_bias else "n"}b_'
                f'{type(self.cost_function).__name__.lower()}_'
                f'{self.batch_size}bs_'
                f'{self.learning_rate}lr_'
                f'{self.momentum}m_'
                f'{self.seed}s_'
                f'{os.path.basename(os.path.splitext(self.train_file)[0])}')

    def perform_test(self):
        train_data = self.__get_train_data(self.train_file)
        test_data = self.__get_test_data(self.test_file)
        train_data_for_evaluation = self.__get_test_data(self.train_file)

        layers = self.hidden_layers + [train_data[0][1].shape[0]]
        layers.insert(0, train_data[0][0].shape[0])
        activation_functions = [self.activation_function] * (len(layers) - 2) + [Identity()]
        visualizer = Visualizer(layers, self.is_bias)
        net = Network(layers,
                      is_bias=self.is_bias,
                      activation_functions=activation_functions,
                      cost_function=self.cost_function,
                      visualizer=visualizer)
        net.train(train_data,
                  iterations=self.iterations,
                  mini_batch_size=self.batch_size,
                  learning_rate=self.learning_rate,
                  momentum=self.momentum,
                  seed=self.seed,
                  iteration_function=self.__run_scoring,
                  iteration_train_data=train_data,
                  iteration_test_data=test_data)

        return self.results

    @staticmethod
    def __get_train_data(filename):
        x, y = pr.load_data(filename)
        x = np.array(x)
        y = np.array(y)
        return [(np.array(_x).reshape(-1, 1), np.array(_y).reshape(-1, 1)) for _x, _y in zip(x, y)]

    @staticmethod
    def __get_test_data(filename):
        x, y = pr.load_data(filename)
        x = np.array(x)
        y = np.array(y)
        return [(np.array(_x).reshape(-1, 1), _y[0]) for _x, _y in zip(x, y)]

    def __run_scoring(self, iteration, iterations, network, train_data, test_data):
        if iteration % self.collect_results_for_iterations:
            return

        test_results = [(network.predict(x), y) for (x, y) in test_data]
        test_err = sum(self.cost_function.function(x, y) for (x, y) in test_results)[0][0]

        train_results = [(network.predict(x), y) for (x, y) in train_data]
        train_err = sum((self.cost_function.function(x, y) for (x, y) in train_results))[0][0]

        self.results += [TestResult(iteration, test_err, train_err)]
        print(f'Iteration {iteration}/{iterations} completed; Test err: {test_err:.2f}; Train err: {train_err:.2f}')


def perform_tests_and_save(tester, results_directory):
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    results = tester.perform_test()
    tv.visualise_test_results(
        results=results,
        title=tester.get_test_title(),
        y_label=tester.get_score_name(),
        save=True,
        filename=results_directory + tester.get_name() + '.png'
    )


def classification_test():
    tester = ClassificationConfiguration(
        train_file="classification/data.three_gauss.train.500.csv",
        test_file="classification/data.three_gauss.test.500.csv",
        iterations=1000,
        hid_layers=[5, 5],
        act_func=ReLU(),
        cost_func=QuadraticCostFunction(),
        is_bias=True,
        batch_size=10,
        lr=0.3,
        moment=0.2,
        seed=123456789,
        collect_results_for_iterations=50
    )

    perform_tests_and_save(
        tester=tester,
        results_directory='tests/results/classification/'
    )


def regression_test():
    tester = RegressionConfiguration(
        train_file="regression/data.activation.train.100.csv",
        test_file="regression/data.activation.test.100.csv",
        iterations=30000,
        hid_layers=[7, 7],
        act_func=ReLU(),
        cost_func=QuadraticCostFunction(),
        is_bias=True,
        batch_size=10,
        lr=0.0001,
        moment=0.01,
        seed=1000,
        collect_results_for_iterations=50
    )

    perform_tests_and_save(
        tester=tester,
        results_directory='tests/results/regression/'
    )


if __name__ == "__main__":
    # classification_test()
    regression_test()
